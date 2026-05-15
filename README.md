# Ollama Paradox Mod Translator

Paradox 게임 MOD의 YML 언어 파일을 로컬 Ollama LLM으로 번역하는 도구

## 사용법

1. Ollama 서버 실행 상태 확인
2. Model / Ollama URL 입력
3. Source(원본 언어), Target(번역할 언어) 선택
4. Input Folder(원본 YML 폴더), Output Folder(출력 폴더) 지정
5. Start Translation 클릭

## 요구사항

- [Ollama](https://ollama.ai) 로컬 서버
- 번역용 LLM 모델 (권장: `translategemma:12b`)
