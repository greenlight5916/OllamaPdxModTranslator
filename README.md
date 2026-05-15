# Ollama Paradox Mod Translator
AI로 제작됨.        
아직 번역탭 말고는 미완성 이므로 기능이 불안정 할 수 있음                         
https://github.com/dltpsk03/pdx_mod_translator 를 기반으로 제작된 Ollama LLM 을 사용하는 번역기                              
(현재 Stellaris만 테스트완료)

## 사용법

1. Ollama 설치 및 번역에 사용할 모델 설치 후 모든 Ollama 프로세스 종료(작업관리자에서 확인)
2. Translator 실행후 Start Ollama버튼 클릭(Ollama 서버실행및 모델 목록 새로고침)
3. 자동으로 모델과 서버URL을 불러오며 모델이 여러개일경우 번역에 사용할 모델 선택
4. Source(원본 언어), Target(번역할 언어) 선택
5. Input Folder(번역할 모드 폴더)및 Output Folder(자동으로 지정되며 수동 지정가능) 경로 지정
6. 게임 프리셋 선택및 번역 세부설정(본인 GPU및 모델에따라 지정)
7. Start Translation 클릭

## 요구사항

- [Ollama](https://ollama.ai) 로컬 서버
- 번역용 LLM 모델 (본인 GPU에 맞는 모델 선택)
