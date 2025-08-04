#!/bin/bash

# 设置脚本名称
PYTHON_SCRIPT="trainer/merger.py" # 假设你的Python脚本名为merger.py

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误：找不到 Python 脚本 '$PYTHON_SCRIPT'。请确保它在当前目录下或更新脚本路径。"
    exit 1
fi

# 默认参数
BASE_MODEL_PATH="" # 现在为必需参数，不再设置默认值
LORA_ADAPTER_PATH=""
OUTPUT_PATH=""
COMPARE_MODELS_FLAG=""
TEST_INPUT_VALUE=""
SAVE_TOKENIZER_FLAG="--save_tokenizer" # 默认保存 tokenizer，对应Python脚本中的 --no_save_tokenizer 逻辑

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --base_model_path)
            BASE_MODEL_PATH="$2"
            shift
            ;;
        --lora_adapter_path)
            LORA_ADAPTER_PATH="$2"
            shift
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift
            ;;
        --compare_models)
            COMPARE_MODELS_FLAG="--compare_models"
            ;;
        --test_input)
            TEST_INPUT_VALUE="--test_input \"$2\"" # 确保字符串中的空格被正确处理
            shift
            ;;
        --no_save_tokenizer) # 对应 Python 脚本中的 --no_save_tokenizer 参数
            SAVE_TOKENIZER_FLAG="--no_save_tokenizer" # 传入这个标志会禁用保存 tokenizer
            ;;
        *)
            echo "未知参数：$1"
            exit 1
            ;;
    esac
    shift
done

# 检查必需参数
if [ -z "$BASE_MODEL_PATH" ]; then
    echo "错误：必须指定 --base_model_path。"
    echo "用法：./run_merger.sh --base_model_path <你的基础模型路径> --lora_adapter_path <你的LoRA路径> --output_path <你的输出路径> [可选参数]"
    exit 1
fi

if [ -z "$LORA_ADAPTER_PATH" ]; then
    echo "错误：必须指定 --lora_adapter_path。"
    echo "用法：./run_merger.sh --base_model_path <你的基础模型路径> --lora_adapter_path <你的LoRA路径> --output_path <你的输出路径> [可选参数]"
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "错误：必须指定 --output_path。"
    echo "用法：./run_merger.sh --base_model_path <你的基础模型路径> --lora_adapter_path <你的LoRA路径> --output_path <你的输出路径> [可选参数]"
    exit 1
fi

# 打印将要执行的命令
echo "正在执行 LoRA 模型合并..."
echo "基础模型路径: $BASE_MODEL_PATH"
echo "LoRA 适配器路径: $LORA_ADAPTER_PATH"
echo "输出路径: $OUTPUT_PATH"
[ -n "$COMPARE_MODELS_FLAG" ] && echo "将进行模型比较"
[ -n "$TEST_INPUT_VALUE" ] && echo "测试输入: $(echo $TEST_INPUT_VALUE | cut -d\" -f2)"
if [ "$SAVE_TOKENIZER_FLAG" == "--save_tokenizer" ]; then
    echo "将保存 tokenizer"
else
    echo "不保存 tokenizer (--no_save_tokenizer)"
fi


# 构建并执行 Python 命令
# 注意：这里我们直接将 $SAVE_TOKENIZER_FLAG 放入命令中，
# 如果它是空字符串，Python脚本会使用其默认值 (save_tokenizer=True)，
# 如果是 "--no_save_tokenizer"，Python脚本则会设置为 False。
CMD="python3 $PYTHON_SCRIPT \
    --base_model_path \"$BASE_MODEL_PATH\" \
    --lora_adapter_path \"$LORA_ADAPTER_PATH\" \
    --output_path \"$OUTPUT_PATH\" \
    $COMPARE_MODELS_FLAG \
    $TEST_INPUT_VALUE \
    $SAVE_TOKENIZER_FLAG"

echo "命令: $CMD"
eval $CMD

# 检查命令执行结果
if [ $? -eq 0 ]; then
    echo "LoRA 合并过程成功完成！"
else
    echo "LoRA 合并过程失败，请检查错误信息。"
    exit 1
fi
