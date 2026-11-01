# PyPI 배포 절차

`pip install newch`로 설치되게 하는 방법입니다. 배포 파일은 `python -m build`로 만들어 `dist/`에 둡니다.

| 파일 | 용도 |
|---|---|
| `newch-<버전>-py3-none-win_amd64.whl` | Windows x86-64. 컴파일 없이 바로 설치됩니다 |
| `newch-<버전>.tar.gz` (sdist) | 그 밖의 플랫폼. 설치할 때 zig(pip 패키지 `ziglang`)로 엔진을 컴파일합니다 |

## 1. 처음 한 번만

1. 계정 만들기: <https://pypi.org/account/register/>, 시험용 <https://test.pypi.org/account/register/> (서로 다른 계정입니다)
2. 두 곳 모두 2단계 인증(2FA)을 켭니다 (업로드에 필요합니다)
3. API 토큰 만들기: 각 사이트의 Account settings → API tokens → "Add API token"
   - 첫 업로드 전에는 프로젝트가 없으므로 범위(scope)를 "Entire account"로 만듭니다
   - 첫 업로드가 끝나면 토큰을 지우고, 범위를 `newch` 프로젝트로 제한한 토큰을 새로 만드세요
4. 토큰은 `pypi-`로 시작합니다. 한 번만 보여 주므로 안전한 곳에 보관하세요

## 2. 배포 파일 만들기

```bash
pip install build twine
rm -rf dist
python -m build            # sdist를 만들고, 그 sdist에서 휠을 다시 빌드합니다
python -m twine check dist/*
```

## 3. TestPyPI에 먼저 올려 보기

```bash
python -m twine upload --repository testpypi dist/*
# Username: __token__
# Password: (TestPyPI 토큰)
```

새 가상환경에서 설치해 봅니다. 소스 설치에 필요한 `setuptools`, `ziglang`은 TestPyPI에 없으므로 진짜 PyPI를 보조 인덱스로 지정합니다.

```bash
python -m venv /tmp/nc-test
/tmp/nc-test/bin/pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ newch
/tmp/nc-test/bin/python -c "import newch as nc; print(nc.__version__, nc.default_context().backend)"
```

## 4. PyPI에 올리기

```bash
python -m twine upload dist/*
# Username: __token__
# Password: (PyPI 토큰)
```

이후 누구나 `pip install newch` (numpy 연동은 `pip install "newch[numpy]"`)로 설치할 수 있습니다.

토큰을 매번 입력하지 않으려면 환경 변수 `TWINE_USERNAME=__token__`, `TWINE_PASSWORD=<토큰>`을 쓰거나 `~/.pypirc`에 저장합니다.
토큰을 저장소에 커밋하지 마세요.

## 5. 다음 버전을 낼 때

1. 버전 번호를 세 곳에서 올립니다: `pyproject.toml`의 `version`, `newch/__init__.py`의 `__version__`, `engine/include/nc.h`의 `NC_VERSION_*`
2. `python tools/build.py`로 전체 테스트를 통과시킵니다
3. 2~4단계를 반복합니다

**주의:** PyPI에서는 한 번 올린 버전 번호를 지워도 **같은 번호로 다시 올릴 수 없습니다.** 문제가 있는 버전은 지우지 말고
PyPI 웹에서 "yank" 처리한 뒤 다음 번호로 올리세요.

## 알아 둘 점

- **sdist 설치는 `ziglang` 패키지(수십 MB)를 내려받아 컴파일합니다.** 저사양 기기에서는 시간이 걸리고 네트워크가 필요합니다.
  리눅스·ARM용 휠을 따로 올리면 이 과정이 없어집니다. ARM 실측이 끝나면 `python tools/build.py --target aarch64` 결과로
  리눅스 휠을 만드는 단계를 추가할 수 있습니다
- **리눅스·macOS에서의 sdist 설치는 아직 실행해 보지 않았습니다.** Windows에서는 휠 설치와 sdist 설치를 새 가상환경에서 모두 확인했고,
  리눅스·macOS에서는 CI가 소스에서 빌드한 엔진으로 전체 테스트를 통과합니다
- PyPI 페이지에는 README.md가 그대로 표시됩니다. README의 상대 링크는 PyPI에서 열리지 않으므로 README 링크는
  GitHub 절대 주소로 씁니다 (`pyproject.toml`의 `[project.urls]`도 GitHub를 가리킵니다)
