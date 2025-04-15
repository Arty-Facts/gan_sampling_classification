import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pathlib

class DB_Logger:
    def __init__(self, db_path="training_results.db", connect_only=False):
        """
        Initialize the logger with a SQLite database.
        
        Args:
            db_path (str): Path to the SQLite database file.
        """
        if isinstance(db_path, str):
            if not db_path.endswith('.db'):
                raise ValueError("Database path should end with '.db'")
            if not pathlib.Path(db_path).parent.exists():
                pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            db_path = pathlib.Path(db_path)
        elif isinstance(db_path, pathlib.Path):
            if not db_path.suffix == '.db':
                raise ValueError("Database path should end with '.db'")
            if not db_path.parent.exists():
                db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("Database path should be a string or pathlib.Path object.")
        if not db_path.exists():
            if connect_only:
                raise FileNotFoundError(f"Database not found at {db_path}")
            print(f"Creating new database at {db_path}")
        else:
            print(f"Connecting to existing database at {db_path}")
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self.conn.execute("PRAGMA synchronous = FULL")

    def _create_tables(self):
        """Create the experiments and results tables if they do not exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                dataset TEXT,
                numb_classes INT,
                seed INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                run_id INTEGER,
                imgs INTEGER,
                acc REAL,
                f1 REAL,
                precision REAL,
                recall REAL,
                confusion_matrix BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assigned_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                run_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')     
        self.conn.commit()

    def register_experiment(self, name, dataset, numb_classes, seed):

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM experiments WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0]



        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO experiments (name, dataset, numb_classes, seed) VALUES (?, ?, ?, ?)",
            (name, dataset, numb_classes, seed)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def get_experiment_ids(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM experiments")
        return [row[0] for row in cursor.fetchall()]
    
    def remove_run(self, experiment_id, run_id):
        self.conn.execute(
            "DELETE FROM results WHERE experiment_id = ? AND run_id = ?",
            (experiment_id, run_id)
        )
        self.conn.commit()
    
    def remove_experiment(self, experiment_id):
        self.conn.execute(
            "DELETE FROM results WHERE experiment_id = ?",
            (experiment_id,)
        )
        self.conn.execute(
            "DELETE FROM experiments WHERE id = ?",
            (experiment_id,)
        )
        self.conn.execute(
            "DELETE FROM assigned_runs WHERE id = ?",
            (experiment_id,)
        )
        self.conn.commit()
    
    def get_experiment_name(self, experiment_id):

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name FROM experiments WHERE id = ?",
            (experiment_id,)
        )
        row = cursor.fetchone()
        return row[0] if row is not None else None

    def get_next_run_id(self, experiment_id):
        ## make sure to read from the database

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT MAX(run_id) FROM assigned_runs WHERE experiment_id = ?",
            (experiment_id,)
        )
        row = cursor.fetchone()
        # make sure that the value is put in the database
        next_id = 0 if row[0] is None else row[0] + 1
        cursor.execute(
            "INSERT INTO assigned_runs (experiment_id, run_id) VALUES (?, ?)",
            (experiment_id, next_id)
        )
        self.conn.commit()
        return next_id

    def report_result(self, experiment_id, run_id, imgs, acc, f1, precision, recall, confusion_matrix):
        self.conn.execute(
            """
            INSERT INTO results 
                (experiment_id, run_id, imgs, acc, f1, precision, recall, confusion_matrix)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (experiment_id, run_id, imgs, acc, f1, precision, recall, confusion_matrix.flatten().astype(np.int32).tobytes())
        )
        self.conn.commit()

    def get_results(self, experiment_id, run_id=None):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT numb_classes FROM experiments WHERE id = ?",
            (experiment_id,)
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"Did not fin experiment_id{experiment_id} in db")
        else:
            numb_classes = row[0]

            # Return all results for the experiment (across runs)
        cursor.execute(
            "SELECT run_id, imgs, acc, f1, precision, recall, confusion_matrix "
            "FROM results WHERE experiment_id = ? ORDER BY run_id, imgs",
            (experiment_id,)
        )

        for row in cursor.fetchall():
            run_id, imgs, acc, f1, precision, recall, confusion_matrix = row
            confusion_matrix = np.frombuffer(confusion_matrix, dtype=np.int32).reshape(numb_classes, numb_classes)
            yield  run_id, imgs, acc, f1, precision, recall, confusion_matrix

    def get_stats(self, experiment_id, metric='f1'):

        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT imgs, {metric} FROM results WHERE experiment_id = ? ORDER BY imgs",
            (experiment_id,)
        )
        data = cursor.fetchall()
        # Group values by imgs
        img_valuess = {}
        for imgs, value in data:
            img_valuess.setdefault(imgs, []).append(value)
        imgs = sorted(img_valuess.keys())
        means = [np.mean(img_valuess[e]) for e in imgs]
        stds = [np.std(img_valuess[e]) for e in imgs]
        maxs = [np.max(img_valuess[e]) for e in imgs]
        samples = [len(img_valuess[e]) for e in imgs]
        return imgs, means, stds, maxs, samples
        
    def get_global_stats(self, experiment_id, metric='f1', ignore_img=100_000):

        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT run_id, imgs, {metric} FROM results WHERE experiment_id = ? ORDER BY run_id",
            (experiment_id,)
        )
        data = cursor.fetchall()
        # Group values by imgs
        run_id_valuess = {}
        for run_id, imgs, value in data:
            run_id_valuess.setdefault(run_id, []).append((imgs, value))
        run_id_best = {}
        for run_id, values in run_id_valuess.items():
            values = sorted(values)
            i, v = zip(*values)
            if i[-1] < ignore_img:
                continue
            run_id_best[run_id] = np.max(v)
        best_values = np.array(list(run_id_best.values()))
        return best_values

    def close(self):
        """Close the SQLite database connection."""
        self.conn.close()


def plot_metric(db, experiment_ids, metric='f1', fig=None, ax=None, prefix=''):
    """
    Plot the mean and standard deviation of a given metric over imgs.
    
    This aggregates the data across all runs of the experiment.
    
    Args:
        experiment_id (int): The experiment identifier.
        metric (str): The metric to plot (e.g., 'f1').
    """
    results = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    for experiment_id in experiment_ids:
        imgs, means, stds, maxs, samples= db.get_stats(experiment_id, metric)
        if len(imgs) == 0:
            continue
        name = db.get_experiment_name(experiment_id)
            


        max_loc = np.argmax(maxs)
        mean_loc = np.argmax(means)
        ax.plot(imgs, means, label=f'{prefix}{name} mean: {means[mean_loc]*100:.1f}±{stds[mean_loc]*100:.1f}', marker='o', alpha=0.7)
        ax.fill_between(imgs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, label='Std Dev')
        color = ax.get_lines()[-1].get_color()
        ax.scatter(imgs[max_loc], maxs[max_loc], color=color, marker='x')
        results.append((prefix+name, means[mean_loc], stds[mean_loc], maxs[max_loc], samples[max_loc]))
        
    ax.set_xlabel('Imgs')
    ax.set_ylabel(metric)
    ax.set_title(f'Macro {metric} over Imgs')
    # add more granular ticks
    ax.grid(True)
    ax.legend()
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def plot_samples(db, experiment_ids, metric='f1', fig=None, ax=None):
    """
    Plot the mean and standard deviation of a given metric over imgs.
    
    This aggregates the data across all runs of the experiment.
    
    Args:
        experiment_id (int): The experiment identifier.
        metric (str): The metric to plot (e.g., 'f1').
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    for experiment_id in experiment_ids:
        imgs, means, stds, maxs, samples= db.get_stats(experiment_id, metric)
        if len(imgs) == 0:
            continue
        name = db.get_experiment_name(experiment_id)
        ax.plot(imgs, samples, label=f'{name} Samples', marker='o', alpha=0.7)
        
    ax.set_xlabel('Imgs')
    ax.set_ylabel('Samples')
    ax.set_title(f'Samples over Imgs')
    # add more granular ticks
    ax.grid(True)
    ax.legend()




if __name__ == '__main__':
    import unittest
    class TestDBLogger(unittest.TestCase):
        def setUp(self):
            self.logger = DB_Logger("test_unittest.db")
            self.exp_id = self.logger.register_experiment(name="test_exp", dataset="dummy", numb_classes=2, seed=123)

        def test_run_id_increment_and_storage(self):
            run_id_0 = self.logger.get_next_run_id(self.exp_id)
            self.assertEqual(run_id_0, 0)
            run_id_1 = self.logger.get_next_run_id(self.exp_id)
            self.assertEqual(run_id_1, 1)

        def test_report_and_get_results(self):
            run_id = self.logger.get_next_run_id(self.exp_id)
            imgs = 10
            acc, f1, precision, recall = 0.8, 0.75, 0.77, 0.74
            cm = np.array([[5, 1], [2, 7]], dtype=np.int32)
            self.logger.report_result(self.exp_id, run_id, imgs, acc, f1, precision, recall, cm)

            results = list(self.logger.get_results(self.exp_id, run_id))
            self.assertEqual(len(results), 1)

            res_run_id, res_imgs, res_acc, res_f1, res_prec, res_rec, res_cm = results[0]
            self.assertEqual(res_run_id, run_id)
            self.assertEqual(res_imgs, imgs)
            self.assertAlmostEqual(res_acc, acc)
            self.assertAlmostEqual(res_f1, f1)
            self.assertAlmostEqual(res_prec, precision)
            self.assertAlmostEqual(res_rec, recall)
            np.testing.assert_array_equal(res_cm, cm)

        def test_multiple_report_and_get_results(self):
            run_id = self.logger.get_next_run_id(self.exp_id)
            results_input = []
            for imgs in [1, 5, 10]:
                acc, f1, precision, recall = np.random.rand(4)
                cm = np.random.randint(0, 10, (2, 2))
                results_input.append((imgs, acc, f1, precision, recall, cm))
                self.logger.report_result(self.exp_id, run_id, imgs, acc, f1, precision, recall, cm)
            results_output = list(self.logger.get_results(self.exp_id, run_id))
            self.assertEqual(len(results_output), len(results_input))
            for i, (res_run_id, imgs, acc, f1, precision, recall, cm) in enumerate(results_output):
                self.assertEqual(res_run_id, run_id)
                self.assertEqual(imgs, results_input[i][0])
                np.testing.assert_array_equal(cm, results_input[i][5])

            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            # # Plot training loss
            # plot_metric(self.logger, self.logger.get_experiment_ids(), metric='f1', ax=ax[0])
            # plot_samples(self.logger, self.logger.get_experiment_ids(), metric='f1', ax=ax[1])
            # # Plot test accuracy
            # plt.tight_layout()
            # plt.savefig("training_results.png")

        def tearDown(self):
            self.logger.close()
            import os
            os.remove("test_unittest.db")
    unittest.main()
