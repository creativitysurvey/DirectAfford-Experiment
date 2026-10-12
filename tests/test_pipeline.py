"""Offline semantic tests. CLIP downloads are replaced by a fixed text encoder.
Synthetic fixtures verify plumbing, not real-world model quality.
"""
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
        return TokenBatch(batch=len(queries))


class TextModel(torch.nn.Module):
    config = types.SimpleNamespace(projection_dim=512)

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(512))

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def forward(self, batch):
        return types.SimpleNamespace(pooler_output=self.anchor.unsqueeze(0).expand(batch,-1))


# Stub only the external text dependency; exercise actual geometry/head/loss.
sys.modules['transformers'] = types.SimpleNamespace(CLIPTextModel=TextModel, CLIPTokenizer=Tokenizer)
from model import DirectAffordModel, AttentionEnhancedGeometricEncoder
from loss import dense_alignment_loss
from metrics import compute_metrics
from real_data import LASORealDataset, sample_points
import train
import evaluate


def small_model(name="offline"):
    model = DirectAffordModel(name)
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
            common=['train.py','--laso-query-mode','affordance','--data-dir',str(data),'--num-points','8','--batch-size','2','--device','cpu','--loss','bce','--output-dir',str(out)]
            with patch.object(train,'DirectAffordModel',small_model), patch.object(sys,'argv',common+['--epochs','1']): train.main()
            with patch.object(train,'DirectAffordModel',small_model), patch.object(sys,'argv',common+['--epochs','2','--resume',str(out/'last.pth')]): train.main()
            checkpoint=torch.load(out/'last.pth',weights_only=False)
            self.assertEqual(checkpoint['epoch'],2)
            fresh=base/'fresh'
            fresh_args=[str(fresh) if x==str(out) else x for x in common]+['--epochs','2']
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
