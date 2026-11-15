#!/usr/bin/env bash
# Export the house from a COPY of the RTAB-Map database, in RTAB-Map's map frame (--opt 2 uses the
# optimized poses already saved in the database, so everything lines up with the saved 2D map).
#
#   export_house.sh /data/twin/isaac/vslam.db /data/twin/isaac/export
#
# Outputs (prefix "house"):
#   house.pgm/.yaml        2D occupancy grid saved in the database (walls for the collision extrusion)
#   house_poses.txt       optimized robot (base_link) poses: stamp x y z qx qy qz qw id
#   house_mesh.obj/.mtl + textures   textured mesh (Poisson surface, multi-texture atlas)
#   house_cloud.ply       coloured point cloud (fallback / inspection)
# Heavy (CPU + several GB RAM): run one at a time and watch free -m.
set -eo pipefail
DB=${1:-/data/twin/isaac/vslam.db}
OUT=${2:-/data/twin/isaac/export}
THREADS=${THREADS:-4}
mkdir -p "$OUT"
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID=69

run() { echo "+ $*"; local t0=$SECONDS; "$@" 2>&1 | grep -v -E "^Processed [0-9]+/" ; echo "  ($((SECONDS - t0)) s)"; }

# 2D grid (house.pgm/.yaml) + poses (fast)
run rtabmap-export --opt 2 --map --poses --poses_format 11 --output house --output_dir "$OUT" "$DB"

# Coloured cloud (--noise_radius crashes this rtabmap 0.23 build on empty node clouds: leave it off)
run rtabmap-export --opt 2 --cloud --decimation 2 --voxel 0.02 --max_range 3.0 --threads "$THREADS" --output house --output_dir "$OUT" "$DB"

# Textured mesh
run rtabmap-export --opt 2 --mesh --texture --texture_size 4096 --texture_count 6 \
    --texture_range 3.0 --texture_blur 50 --decimation 2 --voxel 0.02 --max_range 3.0 \
    --poisson_size 0.03 --max_polygons 600000 \
    --ground_normals_up 0.9 --threads "$THREADS" --output house --output_dir "$OUT" "$DB"

ls -la "$OUT"
