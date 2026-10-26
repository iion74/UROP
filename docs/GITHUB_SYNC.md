# GitHub 동기화

- 저장소: https://github.com/iion74/UROP
- 브랜치: `main`
- 동작: 코드 수정 → 관련 검증 → 작업 파일만 커밋 → 푸시
- 상시 감시 프로그램은 설치하지 않는다. 파일 저장만으로 푸시되지는 않는다.
  Codex는 루트 `AGENTS.md`에 따라 구현 작업을 마칠 때 동기화를 수행한다.

## 최초 인증

GitHub CLI(`gh`)가 없다면 Ubuntu/Debian에서 설치한다.

```bash
sudo apt update
sudo apt install gh
```

이 프로젝트에 쓰기 권한이 있는 GitHub 계정으로 터미널에서 로그인한다.
인증 정보나 토큰을 채팅, 소스 파일, `.env.example`에 넣지 않는다.

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

다른 저장소 전용 SSH 키로 인증된 상태여도 이 저장소 쓰기 권한이
자동으로 부여되지는 않는다.

## 수동 사용

프로젝트 루트에서 변경한 파일을 명시한다.

```bash
bash scripts/sync_github.sh "Update retrieval policy" src/policy.py
```

로그인 전에 만든 로컬 커밋만 올리는 경우:

```bash
git push origin main
```

`git status --short --branch`와 `git log -3 --oneline`으로 상태를 확인한다.
GitHub에 새 변경이 있으면 작업을 보존한 채 차이를 확인한다.
스크립트는 강제 푸시나 자동 충돌 해결을 하지 않는다.

## 업로드 범위

코드, 테스트, 실행 스크립트, README, 동기화 안내와 안전한 환경 설정
예제만 기본 업로드 대상으로 둔다. 실제 `.env`, 가상환경, `data/`,
`paper/`, 모델, 로그, 논문 원본 및 루트의 연구 보고서·퇴고 메모는
업로드하지 않는다. 논문이나 결과 공개는 별도 승인 후 범위를 정한다.

이 설정은 GitHub 소스 동기화이며 웹 서비스 자동 배포 설정은 아니다.
