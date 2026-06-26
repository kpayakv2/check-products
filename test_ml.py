import sys; import traceback; try:
  from src.api.dependencies import get_ml_learning_system
  ml = get_ml_learning_system()
  print(ml.get_model_performance())
except Exception as e:
  traceback.print_exc()
