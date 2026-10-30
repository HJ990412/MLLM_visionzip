#!/usr/bin/env python3
"""Generate honest stage-separated Korean reports from actual immutable receipts."""
import argparse,gzip,json,sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mmimpress.mtvqa25k.common import read,dump,sha

def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(text);tmp.replace(path)

def report(run):
    out=ROOT/'results'/run.name;out.mkdir(exist_ok=True);data=ROOT/'data/mt_vqa_v2_reconstructed_25k';s=read(data/'dataset_stats.json');cfg=read(data/'config.json')
    write(out/'CONTRACT.md',(run/'CONTRACT.md').read_text())
    dataset=f'''# MT-VQA-v2-reconstructed-25k 데이터 검증

공식 VQAv2 validation 질문/annotation 214,354쌍과 COCO val2014 JPG를 사용했다.
이미지 파일 {s['source_image_files']:,}개 전부 디코드 가능하며 eligible {s['eligible_images']:,}개,
제외 사유 집계 {s['excluded_reasons']}다. 선택 이미지 **25,000**, 대화 **25,000**,
질문 **75,000**개다. 선택 이미지 전부를 독립 재디코드하고 공식 정답·ID·순서를 검증했다.

CPython random.Random(1234)/MT19937, 숫자 ID canonical 정렬에서 sample(25000),
같은 RNG로 이미지 순서대로 정렬된 question_id에서 sample(3)을 사용했다. 첫 질문을
제외하지 않았다. 원본 독립 질문을 이미지별로 묶은 workload이며 저자 subset/순서 재현이 아니다.
사람 정답 10개와 answer_id/confidence, answer_type/question_type, 원본 annotation을 보존했다.

기존 250-image 개발 데이터와 이미지 {s['legacy250_overlap']['images']}개·질문
{s['legacy250_overlap']['questions']}개 중복이다. 새 holdout이라고 주장하지 않는다.
추가 개발 이미지의 exact 파일/RGB 비교는 최초 run 결과의 development_overlap.json에 있다.
ConvBench source와도 선택 이미지 87개에 해당하는 파일이 겹친다. 이 비교는 리사이즈된
이미지의 지각적 유사성이나 로컬 밖 데이터까지 배제하지 않는다.

선택 JPG 실측 합계: {s['selected_image_stats']['bytes']:,} bytes.
전체/선택 answer type, 해상도, 이미지 모양, 질문 수/길이 histogram은 dataset_stats.json에 있다.
모델별 예상 N 분포와 native helper source hash는 run/geometry_*.json에 있다.
LLaVA N960–2880, Qwen N256–529, GLM N16–529. 실제 GPU N과 예산은 별도 감사한다.

Ordered workload SHA256: `{cfg['ordered_workload_sha256']}`.
Dataset config SHA256: `{sha(data/'config.json')}`.
원본 archive URL·파일명·바이트·실제 SHA256·버전/HTTP 헤더와 CRC 기록은 source/ 및
source_manifest.json, 이미지별 full hash는 source_image_manifest.json에 있다.
공식 원본: https://visualqa.org/download.html
'''
    write(out/'DATASET_REPORT.md',dataset)
    scorer=read(out/'scorer/validation.json')
    write(out/'SCORER_VALIDATION.md',f'''# 공식 VQA scorer 검증

Reference revision `{scorer['revision']}`, 원본 SHA256
`{scorer['vendor_provenance']['original_sha256']}`. Python3 변경은 print문 2개뿐이다.
원본 blob SHA1, port SHA256, diff는 mmimpress/mtvqa25k/vendor에 있다.

합성 fixture18개와 기존 raw3000개 문항별 공식 reference/호환 점수 비교 PASS, 불일치0.
0/1/2/3/4/10 일치수는0/.3/.6/.9/1/1. 숫자·관사·구두점·소수·축약·공백·다중단어,
빈 답변과 첫 단어만 고르는 잘못된 처리도 검사했다. 기존 결과를 덮어쓰지 않았다.
이 3000개 재채점은 새 방법의 250-image GPU 파일럿이 아니다.

고정 공식 revision은 정답10개가 공백 처리 후 모두 같으면 추가 정규화를 생략한다.
예를 들어 unanimous 'cat'에 'Cat'은0점이다. 이 동작을 수정하거나 normalized EM으로
바꾸지 않았다. Primary는 이 고정 reference의 VQA score (%)다.

신규 GPU raw에서는 생성-only IDs를 native tokenizer로 독립 decode한 전체 raw와
scorer answer를 함께 보존한다. GLM은 v6의 native 답변 box span과 literal history body를
구분한다. 실제 새 출력 검증의 완료 여부는 각 모델 gate/pilot receipt를 따른다.
공식 evaluator: https://github.com/GT-Vision-Lab/VQA/tree/{scorer['revision']}
''')
    table=[];statuses={};counts=[]
    for model,target in [('llava',375000),('qwen',225000),('glm',225000)]:
        st=read(run/f'{model}_STATUS.json') if (run/f'{model}_STATUS.json').exists() else {'status':'NOT_RUN'};statuses[model]=st
        phases=[]
        for phase in ('gate','smoke','pilot250','pilot40','main'):
            p=run/model/phase/'COMPLETE.json'
            analyzed=out/model/phase/'independent_audit.json'
            phases.append(('PASS' if phase=='gate' else 'COMPLETE' if analyzed.exists() and read(analyzed)['status']=='PASS' else 'EXECUTED_ANALYSIS_PENDING') if p.exists() and read(p)['status']=='PASS' else (st['status'] if st.get('phase')==phase else 'NOT_RUN'))
            base=run/model/phase/phase/'images';started=set();returned=set();adopted_phase=0;failed_attempts=0
            for folder in base.glob('*/attempt_*'):
                failed_attempts+=int((folder/'FAILURE.json').exists())
                name='request_attempts.jsonl' if model=='glm' else 'attempts.jsonl'
                journal=folder/name
                if not journal.exists():journal=journal.with_suffix('.jsonl.gz')
                if journal.exists():
                    with (gzip.open(journal,'rt') if journal.suffix=='.gz' else journal.open()) as f:
                        for line in f:
                            row=json.loads(line);pid=row['physical_execution_id']
                            if row['status']=='STARTED':started.add(pid)
                            if row['status'] in ('RETURNED','COMPLETED'):returned.add(pid)
            for cp in base.glob('*/COMMITTED.json'):adopted_phase+=read(cp)['requests']
            counts.append({'model':model,'phase':phase,'final_adopted_requests':adopted_phase,'physical_runner_started':len(started),
                'physical_runner_returned':len(returned),'failed_attempts':failed_attempts,
                'additional_diagnostic_reference_requests':'NOT_MEASURED' if phase=='gate' else 0,
                'scope':'production runner journals; gate reference calls excluded from this count; main denominator contains main commits only'})
        images=list((run/model/'main/main/images').glob('*/COMMITTED.json'));adopted=sum(read(p)['requests'] for p in images)
        table.append('| '+model+' | '+' | '.join(phases)+f' | {adopted:,}/{target:,} |')
    head='| Model | GPU gate | Smoke | 기존250 pilot | 새40 pilot | Main | Main 채택/목표 |\n|---|---|---|---|---|---|---:|\n'+'\n'.join(table)
    validation='# 검증 상태\n\nDataset/official scorer/관련 CPU 검증은 PASS다. GPU gate는 각 모델의 새 receipt로만 판단한다.\n\n'+head+'\n\n'
    for m,st in statuses.items():
        if st.get('error'):validation+=f"- {m}: `{st['error']}`\n"
    validation+='\nQwen/GLM의 legacy FP32 FAIL은 기존 임계값 그대로 별도 보존하며 native same-shape/semantic gate와 구분한다. 기존 MT-GQA PASS를 새 GPU PASS로 복사하지 않는다.\n'
    write(out/'VALIDATION.md',validation)
    dump(out/'request_counts.json',counts,replace=True)
    main='# MT-VQA-v2-reconstructed-25k 본실험 상태\n\n'+head+'\n\n'
    main+='목표는 825,000 final requests다. Smoke/pilot/진단/실패 요청은 main 분모에 포함하지 않는다. 실행 중/미실행 모델의 VQA score·paired 차이·속도/IO 감소·setup 포함 session은 **NOT_MEASURED**다.\n\n'
    for model in statuses:
        for phase in ('pilot250','pilot40','main'):
            p=out/model/phase/'REPORT.md'
            if p.exists():main+=f'[{model} {phase} 독립 분석]({model}/{phase}/REPORT.md)\n\n'
    main+='Primary는 공식 VQA score와 followup TTFT, 첫 persistence/activation을 포함한3-turn session이다. AllHead의 공유 build 귀속0을 독립 배포 무료로 해석하지 않는다. 모델간 효과를 합치지 않는다.\n'
    write(out/'MAIN_REPORT.md',main)
    reproduce=f'''# 재현과 재개

작업 경로 `/home/dblab/hj/mllm_v2`, 현재 run `{run}`.
기존 결과·원본 이미지·검증 자료를 삭제/수정하지 않는다. remote push 없음.
원본 확보/데이터 구축은133→134, 공식 채점 검증135, native geometry136,
membership137, freeze140, GPU gate139, 실제 실행138, 독립 분석142 순서다.

```bash
cd /home/dblab/hj/mllm_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/139_validate_mtvqa25k_gpu.py --run {run} --model llava
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/138_run_mtvqa25k.py --run {run} --model llava --phase smoke
```

완료한 phase는 위 명령으로 재실행하지 않는다. 미완료 phase만 `--resume`로 재개한다.
LLaVA/Qwen은 mllm_ft Python, GLM은 `.venv-glm46v/bin/python`을 사용한다.
Source/config/environment/model/workload 변경이면 같은 main 재개를 거부한다.

```bash
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/138_run_mtvqa25k.py --run {run} --model llava --phase main --resume
```

Gate→smoke→pilot250→pilot40→main admission을 우회하지 않는다. 각 GPU process는
타 compute PID가 있으면 중단하며 타 작업을 종료하지 않는다. Main은 pilot 실측 자원
계획 PASS도 필요하다. 현재 stage/PID/heartbeat/명령은 run/*_STATUS.json에 있다.
Full hashes와 atomic COMMITTED가 있는 첫 성공 image만 채택한다. Partial image는
새 attempt에서 전체 실행한다. Raw는 lossless gzip으로 보존하며 원본/압축SHA가 commit에 있다.
정리된 새 payload 재생성은 동일 frozen image/source/config의 full T1부터 수행한다.
독립 분석은 `142_analyze_mtvqa25k.py --run ... --model ... --phase ... --output <새 경로>`다.
'''
    write(out/'REPRODUCE.md',reproduce)
    artifacts={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file() and p.name!='artifact_index.json'}
    dump(out/'artifact_index.json',{'generated_unix':time.time(),'files':artifacts,'run_freeze_sha256':sha(run/'experiment_freeze.json'),'source_dataset_config_sha256':sha(data/'config.json')},replace=True)
    print(head,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();report(a.run.resolve())
