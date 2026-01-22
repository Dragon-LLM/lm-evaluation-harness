python -m lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B \
    --tasks c3_zho \
    --limit 10

python -m lm_eval \
  --model hf \
  --model_args pretrained=sshleifer/tiny-gpt2 \
  --tasks c3_zho \
  --num_fewshot 0 \
  --limit 200 \
  --device cpu \
  --batch_size 1


python -m scripts.write_out   --output_base_path lm_eval/tasks/linguafin/outputs/cmmlu_preview   --tasks cmmlu   --sets test   --num_fewshot 2   --num_examples 2

python -m scripts.write_out   --output_base_path lm_eval/tasks/linguafin/outputs/xstory_cloze_preview   --tasks xstory_cloze_zh   --sets train   --num_fewshot 2   --num_examples 2

python -m scripts.write_out   --output_base_path lm_eval/tasks/linguafin/outputs/xcsqa_preview   --tasks xcsqa_zh   --sets val   --num_fewshot 2   --num_examples 2

python -m scripts.write_out   --output_base_path lm_eval/tasks/linguafin/outputs/cmrc2018_preview   --tasks cmrc2018   --sets val   --num_fewshot 2   --num_examples 2