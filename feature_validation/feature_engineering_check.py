# Verifica se o dataset de features está correto conforme as orientações dadas
import sys
import pandas as pd
from loguru import logger
from .schema import YamlSchemaLoader
from .validator import FeatureValidator


def feature_engineering_check(schema_path: str, data_path: str) -> int:
    """Exemplo de uso CLI: python -m feature_engineering_check config/schema.yaml data.parquet"""
    loader = YamlSchemaLoader()
    schema = loader.load(schema_path)
    df = pd.read_parquet(data_path)
    validator = FeatureValidator(schema)
    result = validator.validate(df)
    if result.is_clean():
        logger.info("OK: no issues detected.")
    else:
        logger.error("Validation failed.")
        # Opcional: logar detalhes dos issues
        for cat, items in result.as_dict().items():
            if items:
                logger.error(f"{cat}: {len(items)} issue(s)")
                for it in items:
                    logger.error(f"  - {it}")
    return 0 if result.is_clean() else 2


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python feature_engineering_check.py <schema.yaml> <data.parquet>")
        sys.exit(1)
    sys.exit(feature_engineering_check(sys.argv[1], sys.argv[2]))


