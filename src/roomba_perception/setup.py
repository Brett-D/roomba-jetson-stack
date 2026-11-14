from setuptools import setup

package_name = 'roomba_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='brett',
    maintainer_email='brett@example.com',
    description='TensorRT person detection with depth-based distance.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'person_detector = roomba_perception.person_detector:main',
            'ignore_zones = roomba_perception.ignore_zones:main',
            'vslam_watchdog = roomba_perception.vslam_watchdog:main',
            'floor_obstacles = roomba_perception.floor_obstacles:main',
            'nav_coach = roomba_perception.nav_coach:main',
            'explorer = roomba_perception.explorer:main',
            'imu_calibrate = roomba_perception.imu_calibrate:main',
            'ekf_inputs = roomba_perception.ekf_inputs:main',
        ],
    },
)
