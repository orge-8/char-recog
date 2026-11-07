# 来源

- 整图补边、MobileNetV3-Small 二分类及训练选轮逻辑改编自独立 `Mai_recog_self` 角色识别工程；未复制其个人图片、标签库、实验结果或已训练的身份模型。
- ImageNet 底座来自 PyTorch 官方权重地址，文件名与 SHA256 见 README。MobileNetV3 实现由 torchvision 提供。
- 本插件与原来的质量分类工程保持独立，运行数据只写入 SDK 授予的插件数据目录。
- 图片感知哈希去重（`recog/phash.py`，0.7.2 起）参考 `CarefreeSongs712/mohobot` 的
  `mohobot/image_cache.py`（MIT）：借鉴「感知哈希汉明距离 < 5 视为同一张图」的判定
  思想、缓存作用域隔离与可选依赖降级策略；未复制其下载、落盘与 LRU 实现。
  上游版权与许可声明如下：

  ```
  MIT License

  Copyright (c) 2026 CarefreeSongs712

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```
