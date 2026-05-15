# Ollama Paradox Mod Translator

Paradox 게임 MOD의 YML 언어 파일을 로컬 Ollama LLM으로 번역하는 도구 (v0.13)

## 주요 기능

- **YML 파일 번역** — CK3, HOI4, Stellaris, EU4, Vic3, Imperator 등 게임 MOD 언어 파일 일괄 번역
- **게임별 번역 톤** — 각 게임 장르에 맞는 문체 자동 적용
- **게임코드 보호** — `$변수$`, `[스크립트]`, `§X` 색상코드, `£아이콘£`을 Placeholder 치환하여 LLM이 건드리지 못하게 보호
- **구조 보존** — YAML 키/값 구조를 코드에서 직접 관리, LLM은 순수 텍스트 값만 번역
- **Live Output** — 번역 진행 중 원문과 번역문 실시간 비교
- **Validate** — 번역 완료 후 미번역/외국어 혼용/중복 키 검사
- **체크포인트** — 번역 중단 시 자동 저장, 재시작 시 복구
- **Debug Log** — 체크박스 활성화 시 LLM 입출력 데이터 상세 로깅
- **로그 저장** — `rog/` 폴더에 자동 저장

## 사용법

1. Ollama 서버 실행 상태 확인
2. Model / Ollama URL 입력
3. Source(원본 언어), Target(번역할 언어) 선택
4. Input Folder(원본 YML 폴더), Output Folder(출력 폴더) 지정
5. Start Translation 클릭

## 요구사항

- [Ollama](https://ollama.ai) 로컬 서버
- 번역용 LLM 모델 (권장: `translategemma:12b`)
