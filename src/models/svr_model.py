from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from src.models.base import ModelInterface
from src.schema import SECONDARY_FEATURES, REQUIRED_FEATURES

class SVRModel(ModelInterface):
    name = "svr"

    def __init__(self,C=10.0,epsilon=0.01,kernel="rbf",gamma="scale",secondary_features=(),random_state=42,):
        # Hiperparámetros
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        # Columnas secundarias
        self.secondary_features = list(secondary_features)
        self.random_state = random_state

        self.columns_ = None
        self.pipeline_ = None

    def _select_columns(self):
        invalid = set(self.secondary_features) - set(SECONDARY_FEATURES)
        if invalid:
            raise ValueError(f"Columnas secundarias no válidas: {sorted(invalid)}")
        return list(REQUIRED_FEATURES) + list(self.secondary_features)
    
    def train(self, X: DataFrame, y: Series) -> ModelInterface:
        # Columnas elegidas, guardadas para usar las mismas en predict
        self.columns_ = self._select_columns()
        self.pipeline_ = Pipeline([("scaler", StandardScaler()),
        ("svr",SVR(C=self.C,epsilon=self.epsilon,kernel=self.kernel,gamma=self.gamma,),),])
        self.pipeline_.fit(X[self.columns_], y)
        return self
    
    def predict(self, X: DataFrame) -> Series:
        return super().predict(X)
    
