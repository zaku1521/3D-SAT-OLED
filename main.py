from collections import OrderedDict
import test_oled
import test_qm9

oled_config = config = OrderedDict(
    target_columns=['plqy', 'e_ad', 'homo', 'lumo'],  # 'plqy', 'e_ad', 'homo', 'lumo'
    model_name='SAT_OLED_Model',
    data_type='molecule',
    task='multilabel_regression',
    epochs=40,
    learning_rate=0.0002,
    batch_size=8,
    patience=10,
    metrics='r2',
    loss_key='default',
    drop_out=0.5,
    activate_key='default',
    target_normalize='auto',
    pre_norm=True,
    repeat=1,
    lr_type='linear',
    optim_type='AdamW',
    seed=42,
    split_seed=42,
)
qm9_config = OrderedDict(
    target_columns=['homo', 'lumo', 'gap'],  # 'e_ad', 'homo', 'gap'
    model_name='SAT_OLED_Model',
    data_type='molecule',
    task='multilabel_regression',
    epochs=50,
    learning_rate=0.0001,
    batch_size=128,
    patience=20,
    metrics='r2',
    loss_key='default',
    drop_out=0.2,
    activate_key='default',
    target_normalize='auto',
    pre_norm=True,
    repeat=1,
    lr_type='linear',
    optim_type='AdamW',
    seed=42,
    split_seed=42,
)


def main():
    test_oled.main(oled_config)
    test_qm9.main(qm9_config)


if __name__ == "__main__":
    main()
