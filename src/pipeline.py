from .generate_data import generate
from .features import create_feature_store
from .train import train_models,ROOT

def main():
    generate(); create_feature_store(ROOT/'data/omr_scans_all.csv'); metrics=train_models(); print('\nModel comparison\n',metrics.to_string(index=False))
if __name__=='__main__': main()
