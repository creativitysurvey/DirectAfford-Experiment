"""Offline semantic tests. CLIP downloads are replaced by a fixed text encoder.
Synthetic fixtures verify plumbing, not real-world model quality.
"""
import contextlib
import io
import importlib
import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
torch.set_num_threads(1)


class TokenBatch(dict):
    def to(self, device):
        return self


class Tokenizer:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, queries, **kwargs):
        ids = torch.zeros(len(queries),4,dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i,q in enumerate(queries):
            words = q.split()[:4] or ['empty']
            ids[i,:len(words)] = torch.tensor([sum(w.encode())%512 for w in words])
            mask[i,:len(words)] = 1
        return TokenBatch(batch=len(queries),input_ids=ids,attention_mask=mask)


class TextModel(torch.nn.Module):
    config = types.SimpleNamespace(projection_dim=512)

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(512))

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def forward(self, batch, input_ids=None, attention_mask=None):
        tokens = torch.nn.functional.one_hot(input_ids,512).float()
        return types.SimpleNamespace(pooler_output=self.anchor.unsqueeze(0).expand(batch,-1),last_hidden_state=tokens)


# Stub only the external text dependency; exercise actual geometry/head/loss.
sys.modules['transformers'] = types.SimpleNamespace(CLIPTextModel=TextModel, CLIPTokenizer=Tokenizer)
from model import DirectAffordModel, AttentionEnhancedGeometricEncoder
from loss import dense_alignment_loss, loss_components
from selection import CheckpointSelection
from metrics import compute_metrics
from real_data import LASORealDataset, sample_points
import train
import evaluate


def small_model(name="offline",architecture="legacy"):
    model = DirectAffordModel(name,architecture=architecture)
    model.geometric_encoder = AttentionEnhancedGeometricEncoder(num_patches=4,k_neighbors=3,num_layers=1)
    return model


def fixture(base, split, offset=0):
    objects = {str(offset+i): {'points': np.arange(24,dtype=np.float32).reshape(8,3)/24} for i in range(2)}
    annotations = [{'shape_id': str(offset+i), 'affordance': 'grasp', 'mask': [0,1]*4} for i in range(2)]
    base.mkdir(exist_ok=True)
    for name, value in [('objects',objects),('anno',annotations)]:
        with open(base/f'{name}_{split}.pkl','wb') as f:
            pickle.dump(value,f)


class PipelineTests(unittest.TestCase):
    def test_token_attention_masks_padding_and_uses_words(self):
        model = small_model(architecture='query_tokens').eval()
        pcs = torch.randn(2,8,3)
        pooled = torch.nn.functional.normalize(torch.randn(2,512),dim=-1)
        tokens = torch.randn(2,4,512)
        padding = torch.tensor([[False,False,True,True]]*2)
        with torch.no_grad():
            _,original = model.forward_with_logits(pcs,pooled,token_features=tokens,token_padding=padding)
            changed = tokens.clone();changed[:,2:] += 100
            _,masked = model.forward_with_logits(pcs,pooled,token_features=changed,token_padding=padding)
            torch.testing.assert_close(original,masked)
            changed[:,0] *= -1
            _,valid = model.forward_with_logits(pcs,pooled,token_features=changed,token_padding=padding)
            self.assertGreater((valid-original).abs().max().item(),1e-6)
        _,_,logits = model.score_queries(pcs,['grasp handle','open door'])
        logits.sum().backward()
        self.assertGreater(model.token_attention.in_proj_weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in model.text_encoder.parameters()))
        with self.assertRaises(ValueError):
            model.forward_with_logits(pcs,pooled)

    def test_same_object_pairs_and_direction_diagnostic(self):
        spec=importlib.util.spec_from_file_location('diagnose_queries',Path(__file__).resolve().parents[1]/'scripts/diagnose_queries.py')
        diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
        points=np.arange(24,dtype=np.float32).reshape(8,3)
        dataset=types.SimpleNamespace(num_points=6,samples=[
            (points,[0,1]*4,'grasp','a',['grasp?']),
            (points,[1,0]*4,'open','a',['open?']),
            (points,[0,1]*4,'grasp','b',['grasp?'])])
        pcs,y,queries,pairs,eligible=diagnostic.paired_inputs(dataset,8,42)
        self.assertEqual(eligible,1)
        np.testing.assert_array_equal(pcs[0],pcs[1])
        np.testing.assert_array_equal(y[0],1-y[1])
        good=diagnostic.query_diagnostic(y,y)
        self.assertEqual(good['matched_mae'],0)
        self.assertEqual(good['target_weighted_direction_margin'],1)
        wrong=diagnostic.query_diagnostic(y[::-1],y)
        self.assertLess(wrong['swapped_minus_matched_mae'],0)
        neutral=diagnostic.query_diagnostic(np.full_like(y,0.5),y)
        self.assertEqual(neutral['mean_prediction_change'],0)
        dataset.samples=dataset.samples[:1]
        with self.assertRaises(ValueError):diagnostic.paired_inputs(dataset)

    def setUp(self):
        # Fixtures test semantics, not achieved accuracy. Keep fake metrics out
        # of the user-facing unittest output.
        self.capture=contextlib.redirect_stdout(io.StringIO())
        self.capture.__enter__()

    def tearDown(self):
        self.capture.__exit__(None,None,None)

    def test_stage2_pipeline_resumes_without_test_access(self):
        spec=importlib.util.spec_from_file_location('stage2',Path(__file__).resolve().parents[1]/'scripts/run_stage2.py')
        stage2=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage2)
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);data=base/'data';source=base/'source';out=base/'stage2'
            fixture(data,'train',0);fixture(data,'val',10)
            (data/'Affordance-Question.csv').write_text('Object,Affordance,Question0\n')
            for name,file in [('bce','best_joint.pth'),('combined_01','best_joint.pth'),('combined_1','best_auc.pth')]:
                (source/name).mkdir(parents=True);(source/name/file).write_bytes(b'offline-test-checkpoint')
            commands=[]
            def fake_run(command,**kwargs):
                commands.append(command)
                script=Path(command[2]).name
                if script=='evaluate.py':
                    self.assertEqual(command[command.index('--split')+1],'val')
                    dest=Path(command[command.index('--output-dir')+1]);dest.mkdir(parents=True)
                    (dest/'metrics.json').write_text('{}')
                elif script=='diagnose_calibration.py':
                    dest=Path(command[command.index('--output-dir')+1]);dest.mkdir(parents=True)
                    (dest/'diagnostic.json').write_text('{}')
                elif script=='run_structure_screen.py':
                    dest=Path(command[command.index('--output-root')+1]);dest.mkdir(parents=True)
                    (dest/'screen_summary.json').write_text('[]')
                else: self.fail(script)
            argv=['stage2.py','--source-screen',str(source),'--data-dir',str(data),'--output-root',str(out)]
            with patch.object(stage2.subprocess,'run',fake_run),patch.object(sys,'argv',argv):
                stage2.main();stage2.main()
            self.assertEqual(len(commands),7)
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'complete')
            self.assertFalse((data/'anno_test.pkl').exists())

    def test_legacy_score_and_checkpoint_compatibility(self):
        model=small_model(); model.eval()
        pcs=torch.rand(2,8,3); text=torch.nn.functional.normalize(torch.rand(2,512),dim=-1)
        with torch.no_grad():
            embeddings=model(pcs)
            scored,logits=model.forward_with_logits(pcs,text)
        torch.testing.assert_close(scored,embeddings,rtol=0,atol=0)
        torch.testing.assert_close(logits,10*(embeddings*text.unsqueeze(1)).sum(-1))
        small_model().load_state_dict(model.state_dict(),strict=True)
        self.assertFalse(any('query_' in k or 'local_fusion' in k for k in model.state_dict()))

    def test_local_and_query_paths_receive_gradients(self):
        for architecture in ('local','query_local'):
            model=small_model(architecture=architecture)
            pcs=torch.rand(2,8,3); text=torch.nn.functional.normalize(torch.rand(2,512),dim=-1)
            features,logits=model.forward_with_logits(pcs,text)
            _,bce=loss_components(features,text,torch.tensor([[0,1]*4]*2).float(),logits=logits)
            bce.backward()
            self.assertGreater(model.point_local[0].weight.grad.abs().sum().item(),0)
            if architecture=='query_local':
                self.assertGreater(model.query_film.weight.grad.abs().sum().item(),0)
                self.assertGreater(model.query_head[-1].weight.grad.abs().sum().item(),0)
                model.eval()
                with torch.no_grad():
                    f1,l1=model.forward_with_logits(pcs,text)
                    f2,l2=model.forward_with_logits(pcs,-text)
                torch.testing.assert_close(f1,f2,rtol=0,atol=0)
                residual1=l1-10*(f1*text.unsqueeze(1)).sum(-1)
                residual2=l2-10*(f2*(-text).unsqueeze(1)).sum(-1)
                self.assertGreater((residual1-residual2).abs().mean().item(),1e-6)

    def test_query_architecture_train_and_evaluate(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); data=base/'data'; out=base/'run'
            fixture(data,'train',0); fixture(data,'val',10); fixture(data,'test',20)
            argv=['train.py','--data-dir',str(data),'--laso-query-mode','affordance',
                  '--architecture','query_local','--loss','bce','--epochs','1','--device','cpu',
                  '--batch-size','2','--num-points','8','--output-dir',str(out)]
            with patch.object(train,'DirectAffordModel',small_model),patch.object(sys,'argv',argv): train.main()
            eval_args=['evaluate.py','--checkpoint',str(out/'best.pth'),'--data-dir',str(data),
                       '--device','cpu','--output-dir',str(base/'eval')]
            with patch.object(evaluate,'DirectAffordModel',small_model),patch.object(sys,'argv',eval_args): evaluate.main()
            report=json.loads((base/'eval'/'metrics.json').read_text())
            self.assertEqual(report['architecture'],'query_local')
            with np.load(base/'eval'/'predictions.npz') as result:
                self.assertEqual(result['shape_ids'].tolist(),['20','21'])

    def test_token_training_and_real_diagnostic_paths(self):
        spec=importlib.util.spec_from_file_location('query_probe',Path(__file__).resolve().parents[1]/'scripts/diagnose_queries.py')
        diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);data=base/'data';out=base/'run'
            for split,offset in [('train',0),('val',10)]:
                fixture(data,split,offset)
                path=data/f'anno_{split}.pkl'
                rows=pickle.loads(path.read_bytes())
                rows += [{**r,'affordance':'open','mask':[1,0]*4} for r in rows]
                for r in rows:r['class']='door'
                path.write_bytes(pickle.dumps(rows))
            import csv
            with open(data/'Affordance-Question.csv','w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['Object','Affordance']+[f'Question{i}' for i in range(15)])
                writer.writeheader()
                for aff in ['grasp','open']:
                    writer.writerow({'Object':'door','Affordance':aff,**{f'Question{i}':f'{aff} door {i}?' for i in range(15)}})
            argv=['train.py','--data-dir',str(data),'--architecture','query_tokens','--loss','bce',
                  '--epochs','1','--device','cpu','--batch-size','2','--num-points','8','--output-dir',str(out)]
            with patch.object(train,'DirectAffordModel',small_model),patch.object(sys,'argv',argv):train.main()
            common=['diagnose_queries.py','--data-dir',str(data),'--device','cpu','--num-points','8','--max-shapes','1','--steps','1']
            argv=common+['--mode','queries','--checkpoint',str(out/'best.pth'),'--output-dir',str(base/'query')]
            with patch.object(diagnostic,'DirectAffordModel',small_model),patch.object(sys,'argv',argv):diagnostic.main()
            report=json.loads((base/'query/diagnostic.json').read_text())
            self.assertEqual(report['split'],'val')
            self.assertEqual(report['architecture'],'query_tokens')
            self.assertEqual(len(report['selected_pairs']),1)
            argv=common+['--mode','overfit','--architecture','query_tokens','--output-dir',str(base/'probe')]
            with patch.object(diagnostic,'DirectAffordModel',small_model),patch.object(sys,'argv',argv):diagnostic.main()
            report=json.loads((base/'probe/diagnostic.json').read_text())
            self.assertEqual(report['split'],'train')
            self.assertEqual([r['step'] for r in report['history']],[0,1])
            self.assertTrue((base/'probe/overfit_probe.pth').exists())

    def test_query_stage_sequential_resume_and_manifest(self):
        spec=importlib.util.spec_from_file_location('query_stage',Path(__file__).resolve().parents[1]/'scripts/run_query_stage.py')
        stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);data=base/'data';source=base/'stage2';out=base/'stage3'
            fixture(data,'train',0);fixture(data,'val',10)
            (data/'Affordance-Question.csv').write_text('fixture')
            source.mkdir();(source/'status.json').write_text('{"status":"complete"}')
            checkpoint=source/'structure/query_local_bce/best_joint.pth'
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'fixture')
            def child(command,**kwargs):
                if Path(command[2]).name=='diagnose_queries.py':
                    dest=Path(command[command.index('--output-dir')+1]);dest.mkdir()
                    (dest/'diagnostic.json').write_text('{}')
                else:
                    self.assertEqual(Path(command[2]).name,'run_token_screen.py')
                    dest=Path(command[command.index('--output-root')+1]);dest.mkdir()
                    (dest/'screen_summary.json').write_text('[]')
            argv=['run_query_stage.py','--source-stage2',str(source),'--data-dir',str(data),'--output-root',str(out)]
            with patch.object(stage.subprocess,'run',side_effect=child) as launch,patch.object(sys,'argv',argv):
                stage.main();self.assertEqual(launch.call_count,4)
                stage.main();self.assertEqual(launch.call_count,4)
                checkpoint.write_bytes(b'changed')
                with self.assertRaises(ValueError):stage.main()
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'complete')

    def test_calibrated_evaluation_checks_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); data=base/'data'; run=base/'run'
            fixture(data,'train',0);fixture(data,'val',10);fixture(data,'test',20)
            argv=['train.py','--data-dir',str(data),'--laso-query-mode','affordance','--loss','bce',
                  '--epochs','1','--device','cpu','--num-points','8','--batch-size','2','--output-dir',str(run)]
            with patch.object(train,'DirectAffordModel',small_model),patch.object(sys,'argv',argv):train.main()
            from real_data import file_hash
            calibration=base/'calibration.json'
            parameters=dict(slope=0.8,bias=-0.5,source_split='val',checkpoint_sha256=file_hash(run/'best.pth'))
            calibration.write_text(json.dumps(parameters))
            ev=['evaluate.py','--checkpoint',str(run/'best.pth'),'--data-dir',str(data),'--calibration',str(calibration),
                '--device','cpu','--output-dir',str(base/'evaluated')]
            with patch.object(evaluate,'DirectAffordModel',small_model),patch.object(sys,'argv',ev):evaluate.main()
            report=json.loads((base/'evaluated'/'metrics.json').read_text())
            self.assertEqual(report['raw_pooled']['auc'],report['pooled']['auc'])
            self.assertIsNotNone(report['calibration_sha256'])
            parameters['checkpoint_sha256']='wrong';calibration.write_text(json.dumps(parameters))
            ev=[str(base/'mismatch') if v==str(base/'evaluated') else v for v in ev]
            with patch.object(evaluate,'DirectAffordModel',small_model),patch.object(sys,'argv',ev):
                with self.assertRaises(ValueError):evaluate.main()

    def test_calibration_groups_fit_and_rank(self):
        from calibration import split_groups,fit_calibration,apply_calibration
        ids=np.array(['a','a','b','b','c','d'])
        fit,check=split_groups(ids)
        self.assertFalse(set(ids[fit]) & set(ids[check]))
        p=np.array([0.1,0.2,0.3,0.6,0.7,0.8])
        y=np.array([0,0,1,0,1,1])
        parameters=fit_calibration(p,y)
        self.assertLessEqual(parameters['fit_objective'],parameters['identity_objective'])
        q=apply_calibration(p,0.8,-0.5)
        np.testing.assert_array_equal(np.argsort(p),np.argsort(q))
        self.assertEqual(compute_metrics(p,y)['auc'],compute_metrics(q,y)['auc'])
        with self.assertRaises(ValueError): apply_calibration(p,-1,0)

    def test_calibration_cli_rejects_test_input(self):
        spec=importlib.util.spec_from_file_location('diagnose',Path(__file__).resolve().parents[1]/'scripts/diagnose_calibration.py')
        diagnose=importlib.util.module_from_spec(spec); spec.loader.exec_module(diagnose)
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); data=base/'validation'; out=base/'calibration'; data.mkdir()
            (data/'metrics.json').write_text(json.dumps(dict(split='val',dataset='laso',checkpoint_sha256='fixture',calibration_sha256=None)))
            p=np.array([[0.2,0.6]]*6); y=np.array([[0,1]]*6)
            np.savez(data/'predictions.npz',probabilities=p,masks=y,shape_ids=np.array(['a','a','b','b','c','d']))
            argv=['diagnose.py','--predictions-dir',str(data),'--output-dir',str(out)]
            with patch.object(sys,'argv',argv): diagnose.main()
            parameters=json.loads((out/'calibration.json').read_text())
            self.assertFalse(set(parameters['fit_shape_ids']) & set(parameters['check_shape_ids']))
            (data/'metrics.json').write_text(json.dumps(dict(split='test',dataset='laso')))
            with patch.object(sys,'argv',argv):
                with self.assertRaises(SystemExit): diagnose.main()

    def test_joint_selection_does_not_splice_metrics(self):
        selector=CheckpointSelection()
        selector.update(dict(auc=0.90,mae=0.15))
        result=selector.update(dict(auc=0.84,mae=0.08))
        self.assertTrue(result["feasible_auc"])
        self.assertEqual(selector.best_feasible_auc,0.84)
        # Neither checkpoint passes BOTH strict goals.
        result=selector.update(dict(auc=0.88,mae=0.085))
        self.assertTrue(result["joint"])
        self.assertEqual(selector.best_joint,0)
        self.assertFalse(selector.update(dict(auc=None,mae=0.2))["auc"])

    def test_screen_completion_and_no_test_reads(self):
        spec=importlib.util.spec_from_file_location('loss_screen',Path(__file__).resolve().parents[1]/'scripts/run_loss_screen.py')
        screen=importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); data=base/'data'; out=base/'screen'
            fixture(data,'train'); fixture(data,'val',10)
            (data/'Affordance-Question.csv').write_text('Object,Affordance,Question0\n')
            calls=[]
            def fake_run(command,**kwargs):
                calls.append(command)
                child=Path(command[command.index('--output-dir')+1])
                (child/'history.json').write_text(json.dumps([dict(epoch=1,auc=0.88,mae=0.085)]))
            argv=['run_loss_screen.py','--data-dir',str(data),'--output-root',str(out),'--epochs','1']
            with patch.object(screen.subprocess,'run',fake_run),patch.object(sys,'argv',argv):
                screen.main(); screen.main()
            self.assertEqual(len(calls),4)  # completed jobs are not relaunched
            self.assertFalse((data/'anno_test.pkl').exists())
            summary=json.loads((out/'screen_summary.json').read_text())
            self.assertTrue(all(r['both_validation_targets_met'] for r in summary))

    def test_duplicate_run_lock(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d)/'run.lock','a') as handle:
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(sys,'argv',['train.py','--data-dir',d,'--output-dir',d]):
                    with self.assertRaises(RuntimeError): train.main()

    def test_weighted_objective(self):
        p=torch.randn(2,4,3,requires_grad=True); t=torch.randn(2,3)
        masks=torch.tensor([[0,1]*2]*2).float()
        a,b=loss_components(p,t,masks)
        combined=dense_alignment_loss(p,t,masks,alignment_weight=0.1,bce_weight=2)
        torch.testing.assert_close(combined,0.1*a+2*b)
        torch.testing.assert_close(dense_alignment_loss(p,t,masks,'bce'),b)

    def test_padding_and_pairing(self):
        p = np.arange(6).reshape(2,3)
        m = np.array([0,1])
        points,masks = sample_points(p,m,20,np.random.default_rng(4))
        self.assertEqual(points.shape,(20,3))
        np.testing.assert_array_equal((points[:,0] > 0).astype(float),masks)

    def test_missing_object_fails(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); fixture(base,'train')
            with open(base/'objects_train.pkl','wb') as f: pickle.dump({},f)
            with self.assertRaises(KeyError): LASORealDataset(base,query_mode="affordance")

    def test_deterministic_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(Path(d),'val')
            data=LASORealDataset(d,'val',32,42,query_mode='affordance')
            np.testing.assert_array_equal(data[0][0],data[0][0])

    def test_official_questions_and_normalization(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            fixture(base,'val')
            with open(base/'anno_val.pkl','rb') as f:
                annotations=pickle.load(f)
            for a in annotations: a['class']='Knife'
            with open(base/'anno_val.pkl','wb') as f: pickle.dump(annotations,f)
            (base/'Affordance-Question.csv').write_text('Object,Affordance,Question0\nKnife,grasp,Where should I hold the knife?\n')
            dataset=LASORealDataset(base,'val',8)
            points,_,question=dataset[0]
            self.assertEqual(question,'Where should I hold the knife?')
            np.testing.assert_allclose(points.mean(0),0,atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.norm(points,axis=1).max()),1,places=6)

    def test_metrics_undefined_auc_keeps_mae(self):
        result=compute_metrics([0.1,0.2],[0,0])
        self.assertIsNone(result['auc'])
        self.assertAlmostEqual(result['mae'],0.15)
        with self.assertRaises(ValueError): compute_metrics([float('nan')],[0])

    def test_background_gradient_and_empty_alignment(self):
        p=torch.randn(1,3,4,requires_grad=True)
        t=torch.randn(1,4)
        dense_alignment_loss(p,t,torch.zeros(1,3),'bce').backward()
        self.assertGreater(p.grad.abs().sum().item(),0)
        p.grad=None
        dense_alignment_loss(p,t,torch.zeros(1,3),'alignment').backward()
        self.assertEqual(p.grad.abs().sum().item(),0)

    def test_head_updates_text_frozen_eval_deterministic(self):
        model=small_model(); model.train()
        self.assertFalse(model.text_encoder.training)
        self.assertFalse(model.text_encoder.anchor.requires_grad)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        before=model.regression_head[-1].weight.detach().clone()
        pcs=torch.rand(2,8,3)
        loss=dense_alignment_loss(model(pcs),model.get_text_embeddings(['grasp']*2,'cpu'),torch.tensor([[0,1]*4]*2).float(),'bce')
        loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(before,model.regression_head[-1].weight))
        model.eval()
        with torch.no_grad(): torch.testing.assert_close(model(pcs),model(pcs),rtol=0,atol=0)

    def test_train_resume_evaluate(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); data=base/'data'; out=base/'run'
            fixture(data,'train',0); fixture(data,'val',10); fixture(data,'test',20)
            common=['train.py','--laso-query-mode','affordance','--data-dir',str(data),'--num-points','8','--batch-size','2','--device','cpu','--loss','bce','--scheduler','plateau','--lr-patience','0','--selection','joint','--output-dir',str(out)]
            with patch.object(train,'DirectAffordModel',small_model), patch.object(sys,'argv',common+['--epochs','1']): train.main()
            with patch.object(train,'DirectAffordModel',small_model), patch.object(sys,'argv',common+['--epochs','3','--resume',str(out/'last.pth')]): train.main()
            checkpoint=torch.load(out/'last.pth',weights_only=False)
            self.assertEqual(checkpoint['epoch'],3)
            self.assertIn('scheduler',checkpoint)
            self.assertLess(checkpoint['optimizer']['param_groups'][0]['lr'],1e-4)
            for name in ('best_auc.pth','best_mae.pth','best_joint.pth'):
                self.assertTrue((out/name).exists())
            fresh=base/'fresh'
            fresh_args=[str(fresh) if x==str(out) else x for x in common]+['--epochs','3']
            with patch.object(train,'DirectAffordModel',small_model), patch.object(sys,'argv',fresh_args): train.main()
            reference=torch.load(fresh/'last.pth',weights_only=False)
            for name,tensor in checkpoint['model'].items():
                torch.testing.assert_close(tensor,reference['model'][name],rtol=0,atol=0)
            eval_args=['evaluate.py','--checkpoint',str(out/'best.pth'),'--data-dir',str(data),'--output-dir',str(base/'evaluation'),'--device','cpu']
            with patch.object(evaluate,'DirectAffordModel',small_model),patch.object(sys,'argv',eval_args): evaluate.main()
            metrics=json.loads((base/'evaluation'/'metrics.json').read_text())
            self.assertEqual(metrics['samples'],2)
            self.assertTrue((base/'evaluation'/'predictions.npz').is_file())
            fixture(data,'val',11)
            with patch.object(train,'DirectAffordModel',small_model),patch.object(sys,'argv',common+['--resume',str(out/'last.pth')]):
                with self.assertRaises(ValueError): train.main()


if __name__ == '__main__':
    unittest.main()
