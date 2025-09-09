from openai import OpenAI
import json
import time
import re
import argparse
from ase.io import read
from ase import Atoms
from ase_prompt_generator import generate_full_prompt

client = OpenAI(api_key="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")

# 코드블럭 제거 함수
def strip_code_block(text):
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    return match.group(1).strip() if match else text.strip()

# 메시지 출력 함수
def print_message(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
    for msg in messages:
        if msg.role == "assistant":
            return msg.content[0].text.value
    return None

# run 대기
def wait_for_run_completion(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id = thread.id,
            run_id = run.id)
        time.sleep(0.5)
    return run

# Assistant 객체 생성
assisant = client.beta.assistants.create(
    name = "MD Predictor",
    instructions="Predict MD trajectories accurately based on step data and rule constraints. Output JSON only.",
    model = "gpt-4.1",
    tools = [{"type": "file_search"}],
    temperature=0.1
    )
assistant_id = assisant.id
print(f"assistant_id: {assistant_id}")
    
# Vector store + Rule Book 업로드
vector_store = client.vector_stores.create(name="MD_Rulebook_Store")

with open("Rule_book.json","rb") as f:
    up = client.files.create(file=f, purpose="assistants")
print("File.id:", up.id, flush=True)
vs_file = client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=up.id)

client.beta.assistants.update(
    assistant_id = assistant_id,
    tool_resources= {"file_search": {"vector_store_ids": [vector_store.id]}}
)

# Step 불러오기
with open("1-20.json", "r") as f:
    all_data = json.load(f)

# 전체 시작 시간
start_time = time.time()

# Step 예측 loop
for target_step in range(21, 30):
    step_start = time.time()
    # 매 step마다 새로운 thread 생성
    thread = client.beta.threads.create()
    print(f"Thread for step {target_step} created: {thread.id}")

    step_keys = [str(target_step - 3), str(target_step - 2), str(target_step - 1)]
    steps_for_input = {k: all_data[k] for k in step_keys if k in all_data}

    input_json = json.dumps(steps_for_input, indent=2)

    user_prompt = f"""
You are a Computational Chemist specializing in Molecular Dynamics (MD) simulations.
Your expertise is in predicting atomic coordinates based on previous trajectory data while adhering to strict physical and chemical rules.
Your goal is to predict atomic coordinates for future MD steps in a catalyst system while ensuring physical and chemical validity.
Your role is to analyze the provided trajectory data, apply the given rules, and generate accurate atomic coordinates for the target step.
The following is atomic coordinate data for steps {', '.join(step_keys)}:
{input_json}
Based on this, predict atomic coordinates for step {target_step}.
You MUST reference the attached Rule Book using file_search.
Rules:
- Return exactly 195 atoms in the same order as in previous steps.
  GRAPHENE(94), N(4), Fe(1), H2O_H(60), NH_H(2), OH_H(2), H2O_O(30), OH_O(2)
- Format:
{{
  "{target_step}": [
    "GRAPHENE_X+...",
    ...
  ]
}}
- Do not include explanations, markdown, or code block.
- Output only raw JSON.
"""

    # GPT 메시지 전송
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content= user_prompt
    )

    # Run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    # Run 완료 대기
    run = wait_for_run_completion(run, thread)

    # 결과 출력 & 저장
    if run.status == "completed":
        assistant_reply = print_message(thread.id)
        print(assistant_reply)
        assistant_reply = strip_code_block(assistant_reply) #코드블럭 제거
        try:
            parsed_json = json.loads(assistant_reply)
            step_data = parsed_json.get(str(target_step), [])
            # 예측된 결과 저장
            filename = f"prediction_step{target_step:03}_w3.json"
            with open(filename, "w") as f:
                json.dump({str(target_step): step_data}, f, indent=2)
            print(f"Saved {filename} (Atoms: {len(step_data)})") #195개 원자 확인

            # 누적 데이터에 추가
            all_data[str(target_step)] = step_data

            # 다음 입력용 JSON 저장
            if target_step < 550:
                json_filename = f"{target_step - 2}-{target_step}_w3.json"
                next_dict = {
                    str(target_step - 2): all_data[str(target_step - 2)],
                    str(target_step - 1): all_data[str(target_step - 1)],
                    str(target_step): all_data[str(target_step)]
                }
                with open(json_filename, "w") as f:
                    json.dump(next_dict, f, indent=2)

        except json.JSONDecodeError as e:
            print("Failed to parse assistant reply as JSON:", e)    
        
        step_end = time.time()
        print(f"Step {target_step} prediction completed in {step_end - step_start:.2f} seconds")
    
    else:
        print(f"Run failed with status: {run.status}")
        print(run.last_error)

# 전체 종료 시간
end_time = time.time()
print(f"Total time taken for all steps: {end_time - start_time:.2f} seconds")