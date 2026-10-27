"""T06B one-shot runner. Production requires an external qualified lock and authorization."""
from __future__ import annotations
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .t06b_lifecycle import (RunDirectory,RuntimeGuard,ResourceLimitError,canonical,json_read,
                            runtime_fingerprint,safe_path,sha,verify_sources,THREAD_VARIABLES)

IDENTITY="R2C01_METHOD_SYSTEM_DEVELOPMENT_04_T06B"
CONFIG_RELATIVE="configs/development/"+IDENTITY+".json"
SEEDS_RELATIVE="docs/preregistration/R2C01_T06B_SEED_PREFLIGHT.json"
CONFIG_SHA256="55bd916442f55aa1773a3e0f70527c863fd0c6bb9bd25290d24cb0e6a4e0c41e"
SEEDS_SHA256="d2c47271b8e63c54243253b94f6efb7df295a1ca84d4d66d9e036213864a6e0f"

def seed_value(cfg,*tags):
    data=canonical([cfg["base_seed"],cfg["seed_namespace"],*tags])
    return int.from_bytes(hashlib.sha256(data).digest()[:8],"big") & ((1<<63)-1)

def method_order(methods,frame_number):
    n=frame_number%len(methods)
    return methods[n:]+methods[:n]

def verify_authorization(root,cfg,lock_path,authorization_path):
    """No output, generator, estimator or reserved RNG is touched here."""
    root=Path(root).resolve(strict=True)
    if sha(root/CONFIG_RELATIVE)!=CONFIG_SHA256 or cfg!=json_read(root/CONFIG_RELATIVE):
        raise ValueError("registered config changed")
    if lock_path is None or authorization_path is None:
        raise PermissionError("separate execution authorization and qualified source/runtime lock required")
    lock=json_read(lock_path);auth=json_read(authorization_path)
    if (lock.get("identity")!=IDENTITY or lock.get("execution_ready") is not True
            or lock.get("runtime_qualified") is not True):
        raise PermissionError("execution/runtime lock not qualified")
    if (auth.get("status")!="AUTHORIZED_T06B_EXECUTION" or auth.get("identity")!=IDENTITY
            or auth.get("config_sha256")!=CONFIG_SHA256 or auth.get("lock_sha256")!=sha(lock_path)):
        raise PermissionError("authorization is not bound to this experiment")
    if lock.get("config_sha256")!=CONFIG_SHA256 or lock.get("seed_registry_sha256")!=SEEDS_SHA256:
        raise ValueError("lock config/seed mismatch")
    if sha(root/SEEDS_RELATIVE)!=SEEDS_SHA256:raise ValueError("registered seed artifact changed")
    verify_sources(root,lock["inputs"])
    required={p.relative_to(root).as_posix() for p in (root/"src/phase_graph_cme").glob("*.py")}
    required.update({CONFIG_RELATIVE,SEEDS_RELATIVE,"scripts/run_r2c01_t06b.py",
                     "scripts/register_r2c01_t06b_seeds.py",
                     "scripts/run_r2c01_v3_baseline_precision_e0.py",
                     "scripts/validate_r2c01_t06b_run.py",
                     "scripts/validate_r2c01_v3_baseline_precision_e0_03_repair_02.py"})
    if not required.issubset({r["path"] for r in lock["inputs"]}):
        raise ValueError("incomplete execution source closure")
    runtime=runtime_fingerprint()
    if any(runtime["thread_environment"].get(k)!="1" for k in THREAD_VARIABLES):
        raise ValueError("one-thread runtime required")
    if runtime!=lock.get("runtime"):raise ValueError("runtime fingerprint changed")
    pre=lock.get("seed_preflight",{})
    path=safe_path(root,pre.get("path",""))
    if sha(path)!=pre.get("sha256"):raise ValueError("preexecution seed proof unbound")
    report=json_read(path)
    if report.get("identity")!=IDENTITY or report.get("status")!="PASS_PREEXECUTION_METADATA_UNION":
        raise ValueError("fresh seed preflight missing")
    # Rehash declared full live metadata inventory before first sample. New files
    # are also detected by the metadata walker; no scientific package imports.
    import importlib.util
    spec=importlib.util.spec_from_file_location("t06b_seed_metadata",root/"scripts/register_r2c01_t06b_seeds.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,live,_,_,_=module.namespace()["historical_seed_union"](root)
    if list(live)!=report.get("inventory"):
        raise ValueError("preexecution metadata inventory changed")
    if report.get("unexplained_intersection_count")!=0:
        raise ValueError("seed collision not cleared")
    return lock,auth,runtime

def build_tasks(data,cfg,cid,j,f):
    from .t06b_batch import GroupTask
    from .model import PilotGroupModel
    tasks=[]
    for g,(ap,users) in enumerate(zip(data["scatter_ap"],data["scatter_users"],strict=True)):
        pilot=g%cfg["system"]["pilot_symbols"]
        model=PilotGroupModel(data["model_los"][g].copy(),data["model_covariances"][g].copy(),
            data["model_beta"][g].copy(),data["model_noise"][g].copy(),data["model_delta"][g].copy())
        tasks.append(GroupTask(f"AP{int(ap):04d}_PILOT{pilot:04d}",model,
            data["observations"][g].copy(),seed_value(cfg,"v3",cid,j,f,int(ap),pilot)))
    return tasks

def input_arrays(topology,frame,sc):
    from .scenario import pilot_group
    rows=[]
    for ap in range(sc.access_points):
        for p in range(sc.pilot_symbols):
            model,y,users,_=pilot_group(topology,frame,sc,access_point=ap,pilot=p)
            rows.append((ap,model,y,users))
    return dict(channels=frame.channels.copy(),pilot_indices=topology.pilot_indices.copy(),
        ap_positions=topology.ap_positions.copy(),user_positions=topology.user_positions.copy(),
        scatter_ap=np.asarray([x[0] for x in rows],dtype=np.int64),
        scatter_users=np.stack([x[3] for x in rows]),
        model_los=np.stack([x[1].los for x in rows]),
        model_covariances=np.stack([x[1].covariances for x in rows]),
        model_beta=np.stack([x[1].beta for x in rows]),
        model_noise=np.stack([x[1].noise_covariance for x in rows]),
        model_delta=np.stack([x[1].delta for x in rows]),
        observations=np.stack([x[2] for x in rows]))

def scatter_estimates(result,data,cfg):
    s=cfg["system"];estimate=np.empty((s["access_points"],s["antennas_per_ap"],s["users"]),np.complex128)
    covered=np.zeros((s["access_points"],s["users"]),int)
    for g,(ap,users) in enumerate(zip(data["scatter_ap"],data["scatter_users"],strict=True)):
        gid=f"AP{int(ap):04d}_PILOT{g%s['pilot_symbols']:04d}"
        estimate[ap][:,users]=result.estimates[gid];covered[ap,users]+=1
    if not np.all(covered==1) or not np.isfinite(estimate).all():raise ValueError("incomplete scatter")
    return estimate

def verify_replay(results,data,cfg,cid):
    for method,result in results.items():
        for g,(ap,users) in enumerate(zip(data["scatter_ap"],data["scatter_users"],strict=True)):
            gid=f"AP{int(ap):04d}_PILOT{g%cfg['system']['pilot_symbols']:04d}"
            row=next(x for x in result.rows if x["group_id"]==gid)
            if method in ("TRANSFER","TRANSFER_NO_PRIORITY","PHASE","MODEL","CYCLIC","RANDOM"):
                source="V3" if gid in result.selected_ids else "MF"
                if not np.array_equal(result.estimates[gid],results[source].estimates[gid]):
                    raise ValueError("selected/unselected exact replay mismatch")
            if row["status"] in ("B1_FALLBACK","BASELINE_FALLBACK"):
                source="B1" if row["status"]=="B1_FALLBACK" else "B2"
                if not np.array_equal(result.estimates[gid],results[source].estimates[gid]):
                    raise ValueError("fallback mismatch")
    if cid=="NO_LOS":
        for method,result in results.items():
            for row in result.rows:
                gid=row["group_id"]
                if not np.allclose(result.estimates[gid],results["B1"].estimates[gid],rtol=1e-10,atol=1e-10):
                    raise ValueError("no-LoS affine equivalence")
                if method in ("TRANSFER","TRANSFER_NO_PRIORITY") and row["score"]!=0:
                    raise ValueError("no-LoS transfer score")
    return True

def execute_campaign(root,cfg,*,lock_path=None,authorization_path=None,
                     fixture_output=None,fixture_provider=None,fixture_batch=None,
                     fixture_analysis=None,fixture_screen=None):
    """Fixtures must supply all computation; cannot accidentally dispatch real V3."""
    from .receivers import mr_combiner,rzf_combiner
    from .baseline_precision_e0_centered import frame_sufficient_contributions
    root=Path(root).resolve(strict=True)
    fixture=cfg["identity"].startswith("SOFTWARE_FIXTURE_")
    if fixture:
        if not all(x is not None for x in (fixture_output,fixture_provider,fixture_batch,fixture_analysis,fixture_screen)):
            raise PermissionError("fixture must explicitly inject every computation")
        output=Path(fixture_output).resolve()
        if root not in output.parents:raise ValueError("fixture output outside project")
        if output.is_relative_to(root/"results"):raise ValueError("fixtures cannot use scientific results namespace")
        if cfg.get("seed_namespace","").startswith(IDENTITY):raise ValueError("reserved seed namespace forbidden")
        lock=dict(identity=cfg["identity"],execution_ready=False,runtime_qualified=False,inputs=[])
        auth=dict(status="SOFTWARE_FIXTURE_ONLY")
        runtime=runtime_fingerprint()
        batch=fixture_batch;analyze=fixture_analysis;screen=fixture_screen
    else:
        if any(x is not None for x in (fixture_output,fixture_provider,fixture_batch,fixture_analysis,fixture_screen)):
            raise PermissionError("test injection prohibited in real campaign")
        lock,auth,runtime=verify_authorization(root,cfg,lock_path,authorization_path)
        output=safe_path(root,cfg["output_directory"])
        from .t06b_batch import estimate_batch
        from .t06b_statistics import analyze_cell,screen
        batch=estimate_batch;analyze=analyze_cell
    kind="SOFTWARE_FIXTURE" if fixture else "SCIENTIFIC_DEVELOPMENT"
    run=RunDirectory(output,cfg["identity"],kind=kind)
    guard=None
    try:
        run.claim();guard=RuntimeGuard(run.stage,cfg["resource_policy"])
        if fixture:
            for name,obj in (("CONFIG.json",cfg),("LOCK.json",lock),("AUTHORIZATION.json",auth)):
                run.write_json(name,obj)
        else:
            for name,path in (("CONFIG.json",root/CONFIG_RELATIVE),("LOCK.json",Path(lock_path)),("AUTHORIZATION.json",Path(authorization_path))):
                run.write_bytes(name,path.read_bytes())
        run.write_json("RUNTIME.json",runtime)
        if fixture:registry=dict(kind=kind,canonical_seeds={})
        else:registry=json_read(root/SEEDS_RELATIVE)
        if fixture:run.write_json("seed_registry.json",registry)
        else:run.write_bytes("seed_registry.json",(root/SEEDS_RELATIVE).read_bytes())
        shared_startup_seconds=guard.elapsed()
        event_index=0;frame_number=0;groups=0;calls=0;cells=[];offline_total=0.
        def resource_callback():
            from .controller_v3 import V3ResourceLimitError
            try:guard.check()
            except ResourceLimitError as error:raise V3ResourceLimitError(str(error)) from error
        for j in range(cfg["topologies"]):
            geometry=None
            for condition in cfg["conditions"]:
                cid=condition["id"];S=condition["frames_per_topology"]
                generation_start=time.perf_counter()
                if not fixture:
                    from .scenario import ScenarioConfig,generate_topology,generate_channel_frame
                    valid={x.name for x in fields(ScenarioConfig)}
                    sc=ScenarioConfig(**{k:v for k,v in cfg["system"].items() if k in valid},
                        rician_kappa=condition["rician_kappa"],
                        residual_phase_half_width_deg=condition["residual_phase_half_width_deg"])
                    topo=generate_topology(sc,seed=seed_value(cfg,"topology",j))
                    geom=(topo.ap_positions,topo.user_positions,topo.pilot_indices)
                    if geometry is None:geometry=tuple(x.copy() for x in geom)
                    elif not all(np.array_equal(a,b) for a,b in zip(geometry,geom)):
                        raise ValueError("geometry not paired")
                topology_seconds=time.perf_counter()-generation_start
                contributions={m:[] for m in cfg["methods"]};costs={m:0. for m in cfg["methods"]}
                negative_ok=True;accounting_writes=[]
                for f in range(S):
                    guard.check(check_disk=True);frame_start=time.perf_counter()
                    t=time.perf_counter()
                    data=fixture_provider(j,condition,f) if fixture else input_arrays(
                        topo,generate_channel_frame(topo,sc,seed=seed_value(cfg,"frame",cid,j,f)),sc)
                    directory=f"frames/f{frame_number:06d}"
                    input_hash=run.write_npz(directory+"/input.npz",**data)
                    gids=[f"AP{int(ap):04d}_PILOT{g%cfg['system']['pilot_symbols']:04d}" for g,ap in enumerate(data["scatter_ap"])]
                    context=dict(frame_number=frame_number,topology_index=j,condition=cid,frame_index=f)
                    run.write_json(directory+"/input.json",{**context,"group_ids":gids,"input_sha256":input_hash})
                    shared_input=time.perf_counter()-t+topology_seconds/S
                    results={};accounting={};method_wall=[]
                    for method in method_order(cfg["methods"],frame_number):
                        method_start=time.perf_counter();t=time.perf_counter()
                        tasks=build_tasks(data,cfg,cid,j,f);model_seconds=time.perf_counter()-t
                        witness={}
                        def observer(gid,fit):
                            witness[gid]=(fit.fine.z.copy(),fit.coarse.z.copy(),np.stack([x.z for x in fit.all_fits[3:]]))
                        def event(value):
                            nonlocal event_index,calls
                            if value["kind"]=="V3_START":calls+=1
                            run.append_jsonl("events.jsonl",dict(event_index=event_index,**context,**value));event_index+=1
                        event(dict(kind="METHOD_START",method=method))
                        before=hashlib.sha256(b"".join(x.tobytes() for task in tasks for x in
                            (task.model.los,task.model.covariances,task.model.beta,task.model.noise_covariance,task.model.delta,task.observation))).hexdigest()
                        result=batch(tasks,method,fraction=cfg["fraction"],
                            selector_seed=seed_value(cfg,"selector",cid,j,f),frame_index=f,
                            guard=resource_callback,event=event,_mf_observer=observer)
                        after=hashlib.sha256(b"".join(x.tobytes() for task in tasks for x in
                            (task.model.los,task.model.covariances,task.model.beta,task.model.noise_covariance,task.model.delta,task.observation))).hexdigest()
                        if before!=after:raise ValueError("method mutated model input")
                        results[method]=result;groups+=len(result.rows)
                        t=time.perf_counter()
                        estimate=scatter_estimates(result,data,cfg);matrix=estimate.reshape(-1,cfg["system"]["users"])
                        mr=mr_combiner(matrix);rzf=rzf_combiner(matrix,cfg["receivers"]["regularization"])
                        receiver_seconds=time.perf_counter()-t
                        t=time.perf_counter()
                        values=[frame_sufficient_contributions(data["channels"].reshape(matrix.shape),v) for v in (mr,rzf)]
                        c,q,r=(np.stack([x[index] for x in values]) for index in range(3))
                        offline=time.perf_counter()-t;offline_total+=offline
                        arrays=dict(estimate=estimate,mr=mr,rzf=rzf,c=c,q=q,r=r)
                        if witness:
                            if set(witness)!=set(gids):raise ValueError("missing MF witnesses")
                            arrays.update(mf_fine_z=np.stack([witness[g][0] for g in gids]),
                                mf_coarse_z=np.stack([witness[g][1] for g in gids]),
                                mf_start_z=np.stack([witness[g][2] for g in gids]))
                        t=time.perf_counter()
                        run.write_npz(directory+"/"+method+".npz",**arrays)
                        run.write_json(directory+"/"+method+".json",dict(**context,method=method,
                            group_rows=result.rows,selected_ids=list(result.selected_ids),
                            component_seconds=result.component_seconds,input_hash=input_hash))
                        artifact_seconds=time.perf_counter()-t
                        event(dict(kind="METHOD_COMPLETE",method=method))
                        elapsed=time.perf_counter()-method_start;method_wall.append(elapsed)
                        own=elapsed-offline
                        if own<0 or not np.isfinite(own):raise ValueError("invalid cost subtraction")
                        accounting[method]=dict(own_online_seconds=own,offline_evaluation_seconds=offline,
                            estimator_seconds=result.elapsed_seconds,receiver_seconds=receiver_seconds,
                            model_seconds=model_seconds,artifact_seconds=artifact_seconds)
                        contributions[method].append((c,q,r))
                        guard.check(check_disk=True)
                    t=time.perf_counter();negative_ok &= verify_replay(results,data,cfg,cid)
                    offline=time.perf_counter()-t;offline_total+=offline
                    elapsed=time.perf_counter()-frame_start
                    # Shared topology time is amortized, not executed inside this frame.
                    residual=elapsed-(shared_input-topology_seconds/S)-sum(method_wall)-offline
                    if residual < -1e-8:raise ValueError("negative frame accounting residual")
                    overhead=max(0.,residual)
                    for method in cfg["methods"]:
                        accounting[method]["standalone_seconds"]=accounting[method]["own_online_seconds"]+shared_input+overhead/len(accounting)
                        costs[method]+=accounting[method]["standalone_seconds"]
                    t=time.perf_counter()
                    run.write_json(directory+"/accounting.json",dict(**context,shared_input_seconds=shared_input,
                        topology_amortized_seconds=topology_seconds/S,shared_overhead_seconds=overhead,
                        replay_check_seconds=offline,methods=accounting,raw_frame_wall_seconds=elapsed))
                    accounting_write=time.perf_counter()-t
                    accounting_writes.append(accounting_write)
                    for method in cfg["methods"]:
                        costs[method]+=accounting_write/len(cfg["methods"])
                    frame_number+=1
                t=time.perf_counter()
                cqrs=[np.stack([np.stack([item[x] for item in contributions[m]],axis=1) for m in cfg["methods"]]) for x in range(3)]
                analysis=analyze(*cqrs,cfg)
                startup_share=shared_startup_seconds*S/cfg["counts"]["frames"]/len(cfg["methods"])
                for method in cfg["methods"]:costs[method]+=startup_share
                cell=dict(topology_index=j,condition=cid,analysis=analysis,costs=costs,negative_control_pass=bool(negative_ok),
                    shared_accounting_write_seconds=accounting_writes,shared_startup_seconds=shared_startup_seconds,
                    per_method_startup_share_seconds=startup_share,
                    offline_cost_boundary="CELL_NUMERICAL_AND_STATISTICAL_AUDIT_REPORTS_AND_TERMINAL_CERTIFICATION")
                cells.append(cell);run.write_json(f"cells/j{j:03d}_{cid}.json",cell)
                offline_total+=time.perf_counter()-t;guard.check(check_disk=True)
                if analysis["status"]!="PASS":raise FloatingPointError("cell numerical certification failed; diagnostics preserved")
        if not fixture:
            expected=cfg["counts"]
            if (frame_number,groups,calls)!=(expected["frames"],expected["group_method_rows"],expected["real_v3_calls"]):
                raise ValueError("incomplete registry")
        t=time.perf_counter()
        report=screen(cells,cfg,seed_value(cfg,"bootstrap","PRIMARY_TOPOLOGY_VECTOR"))
        run.write_json("SCREEN.json",report);offline_total+=time.perf_counter()-t
        if not fixture:verify_sources(root,lock["inputs"])
        run.write_json("summary.json",dict(identity=cfg["identity"],kind=kind,
            status="COMPLETE_PENDING_INDEPENDENT_VALIDATION",frames=frame_number,group_rows=groups,
            real_v3_calls=0 if fixture else calls,mocked_v3_dispatches=calls if fixture else 0,
            wall_seconds=guard.elapsed(),wall_scope="BEFORE_PUBLICATION_USE_SIBLING_RECEIPT_FOR_TOTAL",
            peak_rss_bytes=guard.peak_rss,artifact_bytes=guard.size(),offline_analysis_seconds=offline_total,
            final_partition="SEALED_UNINSPECTED"))
        run.complete(guard)
        receipt=dict(identity=cfg["identity"],kind=kind,wall_seconds=guard.elapsed(),peak_rss_bytes=guard.peak_rss,
                     artifact_bytes=guard.size(),manifest_sha256=sha(output/"MANIFEST.json"))
        with Path(str(output)+".execution_receipt.json").open("xb") as handle:handle.write(canonical(receipt)+b"\n")
        guard.check(finalizing=True,check_disk=True)
        return receipt
    except BaseException as error:
        from .controller_v3 import V3ResourceLimitError
        run.fail(error,incomplete=isinstance(error,(ResourceLimitError,V3ResourceLimitError)))
        raise
    finally:
        if guard is not None:guard.close()
