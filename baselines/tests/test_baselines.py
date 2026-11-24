import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baselines.model import Model, checkpoint_digest
from baselines.data import flatten, read_dataset
from baselines.predict import predict
from baselines.train import train_run
import numpy as np


class CPUBaselineTests(unittest.TestCase):
    def arrays(self):
        x = np.asarray([[i / 10, (i % 3) / 3] for i in range(24)], dtype=np.float64)
        return x, x[:, 0] * 10 + 2

    def test_resume_preserves_adam_and_matches_continuous_training(self):
        x, y = self.arrays()
        config = dict(hidden=8, seed=42, learning_rate=.01, batch_size=7)
        continuous = Model("regression", ['x', 'y'], [], x, y, **config)
        continuous.fit(x, y, 4)
        resumed = Model("regression", ['x', 'y'], [], x, y, **config)
        resumed.fit(x, y, 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'checkpoint.npz'; resumed.save(path)
            first_digest = checkpoint_digest(path)
            loaded = Model.load(path)
            self.assertEqual(loaded.meta['steps'], 8)
            self.assertTrue(any(np.any(v != 0) for v in loaded.first.values()))
            np.testing.assert_array_equal(loaded.mean, resumed.mean)
            loaded.fit(x, y, 2)
            self.assertEqual(loaded.meta['steps'], continuous.meta['steps'])
            self.assertEqual(loaded.meta['epochs'], 4)
            for key in continuous.params:
                np.testing.assert_array_equal(continuous.params[key], loaded.params[key])
                np.testing.assert_array_equal(continuous.first[key], loaded.first[key])
                np.testing.assert_array_equal(continuous.second[key], loaded.second[key])
            loaded.save(Path(temp) / 'continued.npz')
            self.assertNotEqual(first_digest, checkpoint_digest(Path(temp) / 'continued.npz'))

    def test_checkpoint_bytes_are_reproducible_and_weights_are_cpu_arrays(self):
        x, y = self.arrays()
        with tempfile.TemporaryDirectory() as temp:
            hashes = []
            for i in range(2):
                model = Model('regression', ['x', 'y'], [], x, y, hidden=8)
                model.fit(x, y, 2)
                self.assertEqual(model.meta['device'], 'cpu')
                self.assertLess(model.meta['parameter_count'], 100000)
                self.assertTrue(all(isinstance(v, np.ndarray) for v in model.params.values()))
                path = Path(temp) / f'{i}.npz';model.save(path);hashes.append(checkpoint_digest(path))
            self.assertEqual(hashes[0], hashes[1])

    def test_object_checkpoint_payload_is_never_unpickled(self):
        class Unsafe:
            def __reduce__(self):
                return (eval, ("1 + 1",))
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'bad.npz'
            np.savez(file, payload=np.asarray([Unsafe()], dtype=object))
            with self.assertRaises(ValueError):Model.load(file)

    def test_prediction_checks_feature_schema_and_has_no_label_input(self):
        paths, _ = flatten({'x': 1, 'nested': [2]})
        x, y = self.arrays()
        model = Model('regression', paths, [], x, y, hidden=8)
        self.assertGreaterEqual(predict(model, {'x': 1, 'nested': [2]})['outputs']['value'], 0)
        with self.assertRaises(ValueError):predict(model, {'wrong': 1, 'nested': [2]})
        for key in ('RUL', 'true_rul', 'Normalized_RUL'):
            with self.assertRaises(ValueError):flatten({'nested': {key: 1}})

    def files(self, root, task='regression', labels=None):
        for split in ('train', 'validation'):
            rows = []
            for i in range(12):
                rows.append({'id': f'{split}-{i}', 'inputs': {'x': float(i), 'y': i % 3},
                             'answer': labels[i % len(labels)] if task == 'classification' else i * 2})
            (root / (split + '.jsonl')).write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
        return root / 'train.jsonl', root / 'validation.jsonl'

    def test_only_public_inputs_are_read_and_resume_receipt_binds_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);train, val = self.files(root)
            hidden = root / 'hidden_test.jsonl';hidden.write_text('must not be read')
            original = Path.read_bytes
            def guarded(path):
                if path == hidden:raise AssertionError('hidden labels were read during baseline training')
                return original(path)
            with patch.object(Path, 'read_bytes', guarded):
                first = train_run('C', train, val, root / 'baseline', epochs=1)
                parent = root / 'baseline/checkpoint.npz'
                second = train_run('C', train, val, root / 'round1', resume=parent, epochs=1, stage_id='round1')
            self.assertEqual(second['parent_checkpoint_sha256'], first['checkpoint_sha256'])
            self.assertEqual(second['initial_steps'], first['final_steps'])
            self.assertGreater(second['final_steps'], second['initial_steps'])
            self.assertEqual(second['initial_epochs'], 1)
            self.assertEqual(second['final_epochs'], 2)
            self.assertFalse(second['official_scoring_ready'])
            self.assertEqual(len((root/'round1/history.jsonl').read_text().splitlines()), 1)
            with self.assertRaises(ValueError):train_run('C', train, val, root/'overwrite', resume=parent, epochs=1, seed=43)
            with self.assertRaises(ValueError):train_run('C', train, val, root/'baseline', epochs=1)

    def test_validation_cannot_change_baseline_standardization(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train,val=self.files(root)
            train_run('C',train,val,root/'initial',epochs=1)
            parent=Model.load(root/'initial/checkpoint.npz')
            rows=[json.loads(line) for line in val.read_text().splitlines()]
            for row in rows:row['inputs']['x'] += 1000000
            val.write_text('\n'.join(json.dumps(row) for row in rows))
            train_run('C',train,val,root/'continued',resume=root/'initial/checkpoint.npz',epochs=1)
            continued=Model.load(root/'continued/checkpoint.npz')
            np.testing.assert_array_equal(parent.mean,continued.mean)
            np.testing.assert_array_equal(parent.scale,continued.scale)

    def test_resume_rejects_claimed_architecture_or_seed_change_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train,val=self.files(root)
            train_run('C',train,val,root/'initial',epochs=1)
            parent=root/'initial/checkpoint.npz';digest=checkpoint_digest(parent)
            original=json.loads((Path(__file__).resolve().parents[1]/'configs/C.json').read_text())
            for field,value in [('hidden',64),('seed',43)]:
                config={**original,field:value};profile=root/'changed.json';profile.write_text(json.dumps(config))
                with patch.object(Model,'fit',side_effect=AssertionError('incompatible resume must never train')):
                    with self.assertRaisesRegex(ValueError,'hidden width|random seed'):
                        train_run(profile,train,val,root/'invalid',resume=parent,epochs=1)
                self.assertFalse((root/'invalid').exists())
                self.assertEqual(checkpoint_digest(parent),digest)

    def test_historical_b_classification_requires_explicit_profile_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train,val=self.files(root,'classification',['Normal','Degrading','SevereFault'])
            config=json.loads((Path(__file__).resolve().parents[1]/'configs/B.json').read_text())
            config.update(task='classification',classes=['Normal','Degrading','SevereFault'])
            profile=root/'historical.json';profile.write_text(json.dumps(config))
            with self.assertRaises(ValueError):train_run(profile,train,val,root/'missing',epochs=1)
            provenance=root/'labels.json';provenance.write_text(json.dumps({'status':'draft','method':'test fixture labels; not committee approval'}))
            report=train_run(profile,train,val,root/'baseline',epochs=1,label_provenance=provenance)
            self.assertEqual(report['label_provenance']['declared_status'],'draft')
            model=Model.load(root/'baseline/checkpoint.npz')
            response=predict(model,{'x':1.0,'y':1})
            self.assertEqual({o['label'] for o in response['outputs']},{'Normal','Degrading','SevereFault'})
            self.assertAlmostEqual(sum(o['score'] for o in response['outputs']),1)
            with self.assertRaises(ValueError):read_dataset(train,'classification',['0','1','2'])
            with self.assertRaises(ValueError):train_run('B',train,val,root/'wrong-task',epochs=1)

    def test_b_rul_uses_minutes_and_continues_the_supplied_regression_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train,val=self.files(root)
            report=train_run('B',train,val,root/'baseline',epochs=1)
            self.assertEqual(report['task'],'regression')
            self.assertEqual(report['target_unit'],'minutes')
            self.assertEqual(report['target_origin'],'remaining_time_from_lifecycle')
            self.assertEqual(set(report['validation_after']),{'mae','rmse'})
            self.assertIsNone(report['label_provenance'])
            model=Model.load(root/'baseline/checkpoint.npz')
            self.assertEqual(model.meta['classes'],[])
            self.assertEqual(model.meta['target_unit'],'minutes')
            response=predict(model,{'x':1.0,'y':1})
            self.assertGreaterEqual(response['outputs']['value'],0)
            continued=train_run('B',train,val,root/'continued',resume=root/'baseline/checkpoint.npz',epochs=1)
            self.assertEqual(continued['parent_checkpoint_sha256'],report['checkpoint_sha256'])
            self.assertEqual(continued['initial_steps'],report['final_steps'])
            self.assertGreater(continued['final_steps'],report['final_steps'])
            model.meta['target_unit']='fraction_of_life';model.save(root/'wrong-unit.npz')
            with patch.object(Model,'fit',side_effect=AssertionError('wrong-unit resume must never train')):
                with self.assertRaisesRegex(ValueError,'must use minutes'):
                    train_run('B',train,val,root/'wrong-unit',resume=root/'wrong-unit.npz',epochs=1)

    def test_b_package_delivers_numeric_rul_code_and_weights_without_hidden_data(self):
        from scripts.package_baselines import package
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train,val=self.files(root)
            (root/'hidden_test.jsonl').write_text('private hidden labels')
            report=train_run('B',train,val,root/'baseline',epochs=1)
            manifest=package('b-rul-test',root/'baseline',train,val,root/'packages')
            self.assertEqual(manifest['task'],'regression')
            self.assertEqual(manifest['target_unit'],'minutes')
            self.assertEqual(manifest['label_status'],'approved')
            self.assertEqual(manifest['checkpoint_sha256'],report['checkpoint_sha256'])
            with zipfile.ZipFile(root/'packages/b-rul-test/bundle.zip') as archive:
                self.assertNotIn('labels-provenance.json',archive.namelist())
                self.assertFalse(any('hidden' in path for path in archive.namelist()))
                self.assertEqual(json.loads(archive.read('baselines/configs/B.json'))['task'],'regression')
                self.assertIn('from-baseline',archive.read('START.md').decode())
                self.assertIn('RUL_minutes = (N - i)',archive.read('TASK.md').decode())
                self.assertIn('--resume',archive.read('TRAINING_GUIDE.md').decode())
            self.assertFalse(report['official_scoring_ready'],'approved target definition does not enable formal scoring')

    def test_documented_edition_preserves_trained_code_weights_and_data(self):
        from scripts.package_baselines import enrich_documentation, package
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for key, task, labels, expected in (
                    ('A','classification',['Ethanol','Ethylene','Ammonia','Acetaldehyde','Acetone','Toluene'],'128'),
                    ('B','regression',None,'RUL_minutes = (N - i)'),
                    ('C','regression',None,'RUL_cycles = T - t')):
                train,val=self.files(root,task,labels)
                trained=root/('trained-'+key)
                train_run(key,train,val,trained,epochs=1)
                original=package('original-'+key.lower(),trained,train,val,root/'packages')
                source=root/'packages'/original['id']
                # Repackaging must rely on frozen source, not the checkout's current trainer.
                with patch('scripts.package_baselines.file_sha', wraps=__import__('scripts.package_baselines',fromlist=['file_sha']).file_sha) as hashing:
                    updated=enrich_documentation('documented-'+key.lower(),source,root/'packages')
                self.assertFalse(any(Path(call.args[0]) == Path(__file__).resolve().parents[1]/'train.py' for call in hashing.call_args_list))
                destination=root/'packages'/updated['id']
                for name in ('checkpoint.npz','training.json','history.jsonl','baselines/train.py','public/train.jsonl','public/validation.jsonl'):
                    self.assertEqual((source/name).read_bytes(),(destination/name).read_bytes())
                self.assertEqual(updated['checkpoint_sha256'],original['checkpoint_sha256'])
                self.assertEqual(updated['source_package_sha256'],original['package_sha256'])
                self.assertNotEqual(updated['package_sha256'],original['package_sha256'])
                with zipfile.ZipFile(destination/'bundle.zip') as archive:
                    self.assertIn(expected,archive.read('TASK.md').decode())
                    self.assertIn('TRAINING_GUIDE.md',archive.namelist())
                    self.assertNotIn('manifest.json',archive.namelist())
                (source/'baselines/train.py').write_text('altered training code')
                with self.assertRaisesRegex(ValueError,'integrity|training source'):
                    enrich_documentation('tampered-'+key.lower(),source,root/'packages')

    def test_overlapping_sample_ids_are_not_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train,val=self.files(root);val.write_bytes(train.read_bytes())
            with self.assertRaises(ValueError):train_run('C',train,val,root/'bad',epochs=1)


if __name__ == '__main__':
    unittest.main()
