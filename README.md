# SAPIEN

An easy-to-use virtual avatar platform driven by Large Language Models.

Official code for the paper:

[SAPIEN: Affective Virtual Agents Powered by Large Language Models](https://arxiv.org/abs/2308.03022) 

Masum Hasan, Cengiz Ozel, Sammy Potter, Ehsan Hoque (ACIIW 2023) 


## Demos:
- [Job interview practice](https://www.youtube.com/watch?v=FrV3-n9DbYc)
- [Social conversation](https://www.youtube.com/watch?v=PzWH-5MVJE4)
- [Business pitch](https://www.youtube.com/watch?v=jTgPEXVyn9g)


## How to run

Works on any OS. This fork uses Gemini + ElevenLabs + Whisper (original SAPIEN used Azure).

### 1. Clone and install

```bash
git clone https://github.com/GilJob-E/SAPIEN.git
cd SAPIEN
pip install -r requirements.txt
pip install --upgrade transformers  # sentence-transformers 호환성
```

### 2. API keys 설정

`start_app/dialogue_manager/keys.py`를 생성하고 API 키를 설정:

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"
os.environ["ELEVENLABS_API_KEY"] = "your-elevenlabs-api-key"
os.environ["ELEVENLABS_VOICE_ID"] = "your-voice-id"
```

### 3. Google OAuth 설정

- [Google Cloud Console](https://console.cloud.google.com/apis/credentials)에서 OAuth 2.0 Client ID 생성
- 다운로드한 JSON을 `start_app/client_secret.json`으로 저장
- 승인된 리디렉션 URI에 `http://localhost:5001/callback` 추가

### 4. 설정 파일

```bash
cp start_app/files/local_mode_dummy.json start_app/files/local_mode.json
```

### 5. 아바타 비디오

- Download: https://rochester.box.com/v/sapien-videos
- Place `static` and `speaking` folders under: `start_app/static/video/Metahumans`

### 6. 실행

```bash
cd start_app
TOKENIZERS_PARALLELISM=false python app.py
```

`http://localhost:5001`에서 Google 로그인 후 사용.

### 7. 테스트

```bash
cd start_app
python -m pytest tests/ -v -m "not api"
```

### Tips
- Install `ffmpeg` and add it to Path.
- macOS에서 포트 5000은 AirPlay가 점유하므로 5001 사용.
- `TOKENIZERS_PARALLELISM=false`는 sentence-transformers mutex deadlock 방지에 필수.


## Contributors:
- [Masum Hasan](https://masumhasan.net/)
- [Cengiz Ozel](https://www.cengizozel.com/)
- [Sammy Potter](https://sammypotter.com/)
- [Sara Jeiter-Johnson](https://github.com/josuni)
- Kate Giugno
- Erman Ural
- Richard Chuong

Developed at [Roc-HCI lab](https://roc-hci.com/), University of Rochester
Supervised by, [Prof. Ehsan Hoque](https://hoques.com/)

## Citation
If you use this work, please cite the following paper,

```
@misc{hasan2023sapien,
    title={SAPIEN: Affective Virtual Agents Powered by Large Language Models}, 
    author={Masum Hasan and Cengiz Ozel and Sammy Potter and Ehsan Hoque},
    year={2023},
    eprint={2308.03022},
    archivePrefix={arXiv},
    primaryClass={cs.HC}
}
```

## License

```
MIT License

Copyright (c) 2023 University of Rochester

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.
```

SAPIEN:tm: is a trademark owned by SAPIEN Coach LLC. which is being soft licensed to the University of Rochester. Using the name outside this project is prohibited.
