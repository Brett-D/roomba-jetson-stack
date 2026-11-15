cd /data/twin/isaac/trt
for m in yolo11n yolo11s; do
  /usr/src/tensorrt/bin/trtexec --onnx=$m.onnx --saveEngine=${m}_384x640_fp16.engine --fp16 --memPoolSize=workspace:1024 --warmUp=500 --duration=10 --useSpinWait > ${m}_trtexec.log 2>&1
  grep -E "GPU Compute Time|Throughput|Latency: min" ${m}_trtexec.log | head -4
done
echo TRT_DONE
