# Ollama Paradox Mod Translator

Paradox 게임 MOD의 YML 언어 파일을 로컬 Ollama LLM으로 번역하는 도구

## 주요 기능

- **YML 파일 번역** — Paradox 게임(CK3, HOI4, Stellaris, EU4, Vic3, Imperator) MOD의 언어 파일을 지정한 언어로 일괄 번역
- **게임별 프롬프트** — 각 게임의 분위기에 맞는 번역 톤 자동 적용
- **재시도 및 복구** — API 오류 시 자동 재시도, 중단 시 체크포인트 저장 후 재개 가능
- **Live Output** — 번역 진행 중 원문과 번역문을 실시간으로 좌우 분할 비교
- **Validate** — 번역 완료 후 파일 검사 (미번역 라인, 외국어 혼용, 중복 키 감지)
- **Retry Failed** — 검사에서 발견된 문제 라인을 LLM으로 재번역하거나 수동 수정 후 저장
- **체크포인트** — 번역 중단 시 현재까지의 진행 상황을 자동 저장, 재시작 시 이어서 번역
- **로그 저장** — 모든 로그를 `rog/` 폴더에 자동 저장 (최대 3개 로테이션)

## 사용법

1. Ollama 서버 실행 상태 확인
2. Model / Ollama URL 입력 (Detect 버튼으로 자동 감지)
3. Source(원본 언어), Target(번역할 언어) 선택
4. Input Folder(원본 YML 폴더), Output Folder(출력 폴더) 지정
5. Start Translation 클릭

## 요구사항

- [Ollama](https://ollama.ai) 로컬 서버 실행 중
- 번역할 언어 모델이 Ollama에 설치되어 있어야 함
