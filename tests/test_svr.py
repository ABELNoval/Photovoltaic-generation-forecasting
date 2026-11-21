

from itertools import product

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.example import HourlyMeanModel
from src.models.svr_model import SVRModel
from src.models.utils import postprocess_predictions, split_xy
from src.schema import DAY_FLAG, GHI, SECONDARY_FEATURES, TARGET, TIMESTAMP
from src.data.dataset import load_dataset

# def score(model, val_df):
#     """MAE y RMSE sobre las 24 horas, con el mismo postproceso que la evaluación."""
#     X_all, y_all = split_xy(val_df, daytime_only=False)
#     is_day = val_df[DAY_FLAG]
#     pred = postprocess_predictions(model.predict(X_all), is_day)
#     mae = mean_absolute_error(y_all, pred)
#     rmse = float(np.sqrt(mean_squared_error(y_all, pred)))
#     return mae, rmse
def score(model, val_df):
    """MAE y RMSE sobre las 24 horas y solo sobre las horas de luz."""
    X_all, y_all = split_xy(val_df, daytime_only=False)
    is_day = val_df[DAY_FLAG]
    pred = postprocess_predictions(model.predict(X_all), is_day)
    day = (is_day == 1).to_numpy()

    def rmse(y, p):
        return float(np.sqrt(mean_squared_error(y, p)))

    return {
        "mae": mean_absolute_error(y_all, pred),
        "rmse": rmse(y_all, pred),
        "mae_day": mean_absolute_error(y_all[day], pred[day]),
        "rmse_day": rmse(y_all[day], pred[day]),
    }


def fmt(s):
    return (f"MAE {s['mae']:.4f}  RMSE {s['rmse']:.4f} | "
            f"de día: MAE {s['mae_day']:.4f}  RMSE {s['rmse_day']:.4f}")


def check_alignment(df):
    """Correlación potencia-GHI con distintos desfases (el máximo debe estar en 0)."""
    day = df[df[DAY_FLAG] == 1]
    for lag in [-2, -1, 0, 1, 2]:
        print(f"lag {lag:+d}: {day[TARGET].corr(day[GHI].shift(lag)):.3f}")


def tune(train_df, val_df):
    X_tr, y_tr = split_xy(train_df)  

    print("== Alineación potencia / GHI ==")
    check_alignment(train_df)

    
    base = HourlyMeanModel().train(X_tr, y_tr)
    print("\n== Línea base (media por hora) ==")
    print(fmt(score(base, val_df)))

    
    print("\n== Rejilla C / epsilon / gamma ==")
    results = []
    for C, eps, gamma in product([1, 3, 10, 30], [0.01], [0.0003, 0.001, 0.003]):
        m = SVRModel(C=C, epsilon=eps, gamma=gamma).train(X_tr, y_tr)
        s = score(m, val_df)
        results.append((s["mae_day"], s["rmse_day"], C, eps, gamma))
        print(f"C={C:<4} eps={eps:<6} gamma={gamma!s:<6} {fmt(s)}")
    mae, rmse, C, eps, gamma = min(results)
    print(f"\nMejor: C={C}, epsilon={eps}, gamma={gamma} (MAE {mae:.4f})")

    
    print("\n== Columnas secundarias ==")
    chosen, best_mae = [], mae
    improved = True
    while improved:
        improved = False
        candidates = [c for c in SECONDARY_FEATURES if c not in chosen]
        trials = []
        for col in candidates:
            m = SVRModel(C=C, epsilon=eps, gamma=gamma, secondary_features=chosen + [col])
            m.train(X_tr, y_tr)
            trials.append((score(m, val_df)["mae_day"], col))
        if trials:
            trial_mae, col = min(trials)
            
            if trial_mae < best_mae:
                chosen.append(col)
                best_mae = trial_mae
                improved = True
                print(f"+ {col}: MAE {trial_mae:.4f}")
    print("Secundarias elegidas:", chosen or "ninguna")

     
    final = SVRModel(C=C, epsilon=eps, gamma=gamma, secondary_features=chosen)
    final.train(X_tr, y_tr)
    print("\n== Final ==")
    print(fmt(score(final, val_df)))
    return final

def main():
    df=load_dataset("data/sample.csv")
    train_df = df.loc[:"2014-04-30"]
    val_df = df.loc["2014-05-01":"2014-06-01"]
    best_model = tune(train_df, val_df)
    

if __name__ == "__main__":
    main()