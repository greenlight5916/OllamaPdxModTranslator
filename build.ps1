pyinstaller --distpath . --workpath build\tmp --noconfirm OllamaTranslator.spec
if ($?) { Remove-Item -Recurse -Force build\tmp, build\OllamaTranslator -ErrorAction SilentlyContinue }
