"""Review candidate: verify fixtures by default; explicitly request ONE NPU call.

The NPU mode is prepared and CPU-mocked, not itself hardware-validated. It uses
the already captured owned fixtures, a hash-pinned installed provider and an
explicit matching transaction archive. No SDK files are changed.
"""
from __future__ import annotations

import argparse
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

DLL_SHA = "7e77ad4a8a339195d50c8bbc383635e502f7f6c1034f704af4a2f64b1fb5827f"
TXN_SHA = "f83212a6eb31cea75700d83b1d066c3b5f84960c4e070b05c982406f85520e1b"
NAMES = ("query", "key", "value", "state", "log_gate", "beta")
SHAPES = ((1,64,2048),(1,64,2048),(1,64,4096),(1,32,128,128),(1,64,32),(1,64,32))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write(path, value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")


def npz(path):
    import numpy as np
    require(path.stat().st_size < 16*1024**2,"Fixture archive too large")
    with zipfile.ZipFile(path) as archive:
        require(sum(x.file_size for x in archive.infolist()) < 32*1024**2,"Fixture arrays too large")
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}


def verify(root, case):
    import numpy as np
    fixtures_path=Path(__file__).resolve().with_name("repro-fixtures.json")
    fixtures=json.loads(fixtures_path.read_text())
    row=fixtures["cases"][case]
    paths={}
    for role,info in row["artifacts"].items():
        p=(root/info["path"]).resolve()
        require(p.is_relative_to(root),"Fixture path escapes evidence directory")
        require(p.is_file() and p.stat().st_size==info["bytes"] and sha(p)==info["sha256"],f"Changed {role}")
        paths[role]=p
    raw=npz(paths["inputs"])
    require(set(raw)==set(NAMES),"Unexpected input names")
    for name,shape in zip(NAMES,SHAPES,strict=True):
        v=raw[name]
        require(v.dtype==np.uint16 and v.shape==shape,"Expected fixed BF16 input")
        decoded=(v.astype(np.uint32)<<16).view(np.float32)
        require(bool(np.isfinite(decoded).all()),"Nonfinite input fixture")
    return row,paths,raw


def own_snapshot(executable, output):
    # Only aggregate our own contexts. Never retain other process/device data.
    with tempfile.TemporaryDirectory(prefix="xrt-query-",dir=output) as tmp:
        path=Path(tmp)/"snapshot.json"
        proc=subprocess.run([str(executable),"--batch","examine","-r","aie-partitions","-f","JSON","-o",str(path)],
               capture_output=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
        require(proc.returncode==0 and path.is_file(),"XRT counter query failed")
        data=json.loads(path.read_text(encoding="utf-8-sig"))
    total={"command_submissions":0,"command_completions":0,"errors":0}
    other_pending=0
    for device in data.get("devices",[]):
        for partition in device.get("aie_partitions",{}).get("partitions",[]):
            for c in partition.get("hw_contexts",[]):
                if int(c["pid"])==os.getpid():
                    for k in total:
                        total[k]+=int(c[k])
                else:
                    other_pending+=max(0,int(c["command_submissions"])-int(c["command_completions"]))
    return {"own":total,"other_pending_commands":other_pending}


def windows_limits():
    """2GiB process commit cap; parent separately enforces180 seconds."""
    class Basic(ct.Structure):
        _fields_=[("process_time",ct.c_int64),("job_time",ct.c_int64),("flags",wt.DWORD),
                  ("min_ws",ct.c_size_t),("max_ws",ct.c_size_t),("active",wt.DWORD),
                  ("affinity",ct.c_size_t),("priority",wt.DWORD),("schedule",wt.DWORD)]
    class IO(ct.Structure):
        _fields_=[(f"v{i}",ct.c_uint64) for i in range(6)]
    class Extended(ct.Structure):
        _fields_=[("basic",Basic),("io",IO),("process_memory",ct.c_size_t),("job_memory",ct.c_size_t),
                  ("peak_process",ct.c_size_t),("peak_job",ct.c_size_t)]
    k=ct.WinDLL("kernel32",use_last_error=True)
    k.GetCurrentProcess.restype=wt.HANDLE
    k.CreateJobObjectW.argtypes=[ct.c_void_p,wt.LPCWSTR];k.CreateJobObjectW.restype=wt.HANDLE
    k.SetInformationJobObject.argtypes=[wt.HANDLE,ct.c_int,ct.c_void_p,wt.DWORD]
    k.AssignProcessToJobObject.argtypes=[wt.HANDLE,wt.HANDLE]
    k.SetPriorityClass.argtypes=[wt.HANDLE,wt.DWORD]
    k.GetProcessAffinityMask.argtypes=[wt.HANDLE,ct.POINTER(ct.c_size_t),ct.POINTER(ct.c_size_t)]
    k.SetProcessAffinityMask.argtypes=[wt.HANDLE,ct.c_size_t]
    current=k.GetCurrentProcess();old,system=ct.c_size_t(),ct.c_size_t()
    require(k.GetProcessAffinityMask(current,ct.byref(old),ct.byref(system)),"Cannot read CPU affinity")
    require(k.SetProcessAffinityMask(current,old.value & -old.value),"Cannot restrict CPU affinity")
    require(k.SetPriorityClass(current,0x4000),"Cannot set BelowNormal")
    job=k.CreateJobObjectW(None,None);require(job,"Cannot create JobObject")
    limit=Extended();limit.basic.flags=0x100;limit.process_memory=2*1024**3
    require(k.SetInformationJobObject(job,9,ct.byref(limit),ct.sizeof(limit)),"Cannot set memory cap")
    require(k.AssignProcessToJobObject(job,current),"Cannot assign memory cap")
    return job  # retain handle for complete worker lifetime


def run_one(ort, model, raw, dll, cache, token_backend, snapshot, report):
    """Dependency injection permits CPU mocks; exactly one construction/run."""
    import numpy as np
    ort.register_execution_provider_library("RyzenAILightExecutionProvider",str(dll))
    devices=[d for d in ort.get_ep_devices() if d.ep_name=="RyzenAILightExecutionProvider" and d.device.type==ort.OrtHardwareDeviceType.NPU]
    require(len(devices)==1,"Exactly one NPU device required")
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    provider={"dd_cache":cache.as_posix(),"onnx_custom_ops_const_key":"","compile_fusion_rt":"1"}
    if token_backend:
        provider["hybrid_opt_token_backend"]="npu"
    options.add_provider_for_devices(devices,provider)
    options.register_custom_ops_library(str(dll))
    report["session_constructions_started"]=1
    session=ort.InferenceSession(str(model),sess_options=options,enable_fallback=0)
    session.disable_fallback()
    report["after_session"]=snapshot()
    require(report["after_session"]["other_pending_commands"]==0,"Another NPU workload is active")
    report["providers"]=session.get_providers()
    outputs={"attention":np.full((1,64,4096),0x7fc1,np.uint16),"state_out":np.full((1,32,128,128),0x7fc1,np.uint16)}
    bind=session.io_binding()
    for name,array in raw.items():
        bind.bind_input(name,"cpu",0,16,array.shape,array.ctypes.data)
    for name,array in outputs.items():
        bind.bind_output(name,"cpu",0,16,array.shape,array.ctypes.data)
    bind.synchronize_inputs()
    report["execute_calls_started"]=1
    try:
        session.run_with_iobinding(bind)
    finally:
        report["after_execute"]=snapshot()
    bind.synchronize_outputs()
    return outputs


def worker(args):
    require(os.name=="nt","Native runner requires Windows")
    job=windows_limits()
    row,paths,raw=verify(args.evidence_root,args.case)
    require(sha(args.provider_dll)==DLL_SHA and sha(args.txn_archive)==TXN_SHA,"Installed runtime/transactions do not match evidence")
    require(args.output.is_dir() and not (args.output/'report.json').exists(),"Expected fresh parent-created output")
    report={"status":"failed","session_constructions_started":0,"execute_calls_started":0,"retries":0,
            "provider_dll_sha256":DLL_SHA,"txn_archive_sha256":TXN_SHA,"runner_sha256":sha(__file__),
            "xrt_smi_sha256":sha(args.xrt_smi),"case":args.case,"limits":{"seconds":180,"process_commit_bytes":2*1024**3,"logical_cpus":1},
            "native_runner_qualification":"This new runner has CPU mocks; original saved observations used a separately pinned runner."}
    start=time.monotonic();handles=[]
    try:
        cache=args.output/'cache';cache.mkdir()
        for src,dst in [(paths['model'],args.output/'model.onnx'),(args.txn_archive,cache/'txn_bins.zip')]:
            with src.open('rb') as s,dst.open('xb') as d:shutil.copyfileobj(s,d,1024*1024)
        os.chdir(args.output)
        handles.append(os.add_dll_directory(str(args.provider_dll.parent)))
        handles.append(ct.CDLL(str(args.provider_dll)))
        import onnxruntime as ort
        report['ort_version']=ort.__version__
        snapshot=lambda:own_snapshot(args.xrt_smi,args.output)
        report['before_session']=snapshot()
        require(report['before_session']['other_pending_commands']==0,'Another NPU workload is active')
        before={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in raw.items()}
        # Original measured runner registers the provider from the DLL directory.
        original_register=ort.register_execution_provider_library
        class RuntimeView:
            def __getattr__(self,name):return getattr(ort,name)
            def register_execution_provider_library(self,*values):
                try:
                    os.chdir(args.provider_dll.parent);return original_register(*values)
                finally:os.chdir(args.output)
        outputs=run_one(RuntimeView(),args.output/'model.onnx',raw,args.provider_dll,cache,row['token_backend'],snapshot,report)
        import numpy as np
        with (args.output/'observed_bf16.npz').open('xb') as f:np.savez_compressed(f,**outputs)
        report['output_sha256']=sha(args.output/'observed_bf16.npz')
        refs=npz(paths['reference']);report['metrics']={}
        for name,v in outputs.items():
            with np.errstate(invalid='ignore'):
                actual=(v.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
            m={'elements':v.size,'finite':int(np.isfinite(actual).sum()),'nan':int(np.isnan(actual).sum())}
            if m['finite']==m['elements']:
                error=actual-refs[name]
                m['relative_rmse']=float(np.sqrt(np.mean(error**2))/np.sqrt(np.mean(refs[name]**2)))
            report['metrics'][name]=m
        require(before=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in raw.items()},'Input mutated')
        delta={k:report['after_execute']['own'][k]-report['after_session']['own'][k] for k in report['after_execute']['own']}
        require(delta=={'command_submissions':row['npu_commands'],'command_completions':row['npu_commands'],'errors':0},'Unexpected NPU counters')
        report['npu_delta']=delta
        report['status']='observed_finite' if all(v['finite']==v['elements'] for v in report['metrics'].values()) else 'observed_nonfinite'
        verify(args.evidence_root,args.case)
    except Exception as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        write(args.output/'report.json',report)
        require(bool(job) and len(handles)>=0,'Missing memory guard')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--case',required=True,choices=['batched_m4','batched_m128','loop_m4','loop_m128','no_cast_hints_m128'])
    p.add_argument('--execute-npu',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--provider-dll',type=Path);p.add_argument('--txn-archive',type=Path);p.add_argument('--xrt-smi',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();args.evidence_root=args.evidence_root.resolve()
    row,_,_=verify(args.evidence_root,args.case)
    if not args.execute_npu:
        require(not args.worker,'Worker requires explicit execution flag')
        print(json.dumps({'status':'fixture_verified_cpu_only','case':args.case,'npu_calls':0,'expected_npu_commands':row['npu_commands']}));return
    for k in ('provider_dll','txn_archive','xrt_smi','output'):
        require(getattr(args,k) is not None,f'Missing {k}');setattr(args,k,getattr(args,k).resolve())
    if args.worker:
        worker(args);return
    require(os.name=='nt','Native runner requires Windows')
    require(sha(args.provider_dll)==DLL_SHA and sha(args.txn_archive)==TXN_SHA,'Provider/transaction SHA mismatch')
    args.output.mkdir(exist_ok=False)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--execute-npu','--case',args.case]
    for k in ('evidence_root','provider_dll','txn_archive','xrt_smi','output'):command.extend(['--'+k.replace('_','-'),str(getattr(args,k))])
    start=time.monotonic();code=None;timed_out=False
    with (args.output/'worker.log').open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:code=child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timed_out=True;child.kill();code=child.wait()
    write(args.output/'supervisor.json',{'child_exit_code':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-start,
        'clean_process_exit':code==0 and not timed_out,'qualification':'A saved worker report alone does not establish clean process exit.'})
    require(code==0 and not timed_out,'Worker did not exit cleanly; do not label this a successful run')


if __name__=='__main__':
    main()
