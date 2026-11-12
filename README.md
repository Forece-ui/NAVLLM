# NAVLLM

NAVLLM 是一个自适应码率（ABR）视频实验项目，包含视频发送与接收、NAVLLM / BBA / MPC 策略对比，以及用于查看吞吐量、缓冲、码率和 chunk 日志的网页控制台。

**[打开网页演示](https://forece-ui.github.io/NAVLLM/)**

演示使用示例数据填充控制台和结果汇总。视频页的三个播放器使用同一段生成的动画，NAVLLM 播放器还支持加载允许跨域访问的 HTTPS HLS 地址。下载后也可以直接在浏览器中打开 `index.html` 查看演示。

## 运行本地控制台

在项目根目录运行以下命令，然后访问 [本地控制台](http://127.0.0.1:8088/)。

```sh
python web_controller.py --host 127.0.0.1 --port 8088
```

完整实验需要按控制台表单填写 NAVLLM 模型目录与经验池，准备与模型兼容的 PyTorch、Transformers、NumPy、Munch 等 Python 依赖，并在发送端安装 FFmpeg 和 ffprobe。模型权重、经验池和原始视频需自行准备。

点击「启动三路接收」后，用本地 MP4 发送测试视频。以下示例让发送端使用本机默认的三个接收端口：

```sh
python video_sender.py --input example.mp4 --server-host 127.0.0.1 --ports 5001 5002 5003 --limit-mbps 0
```

「实验结果」页嵌入独立的图表服务，默认端口为 `5173`。GitHub Pages 上的页面使用静态结果汇总。

## 来源与许可

- ABR 基线和模型代码源自 [NetLLM](https://github.com/duowuyms/NetLLM)，其 MIT 声明保留在 [LICENSE](LICENSE) 中。
- 内置播放器脚本来自 [hls.js v1.6.13](https://github.com/video-dev/hls.js/tree/v1.6.13)，见 [版权声明](licenses/hls.js.txt) 和 [Apache 2.0 许可证](licenses/Apache-2.0.txt)。
