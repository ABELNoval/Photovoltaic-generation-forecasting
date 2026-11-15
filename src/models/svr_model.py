from pandas import DataFrame, Series
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
        return super().train(X, y)
    
    def predict(self, X: DataFrame) -> Series:
        return super().predict(X)
    
