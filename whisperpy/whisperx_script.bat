@echo on


if "%1"=="" (
  echo 错误:缺少输入音频文件参数 
  exit /b 1
)

set start_time=%time% 

set input_file=%1

whisperX.exe %input_file% --model large-v3 --diarize --hf_token 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 --max_speakers 5 --language zh

set end_time=%time%

REM 计算脚本执行时间 
set /A elapsed_time=(%end_time:~0,2%*3600 + %end_time:~3,2%*60 + %end_time:~6,2%*1) - (%start_time:~0,2%*3600 + %start_time:~3,2%*60 + %start_time:~6,2%*1)

echo 脚本执行时间为:%elapsed_time% 秒


 