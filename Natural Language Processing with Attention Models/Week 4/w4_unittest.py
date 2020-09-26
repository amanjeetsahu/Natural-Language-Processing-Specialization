import numpy as np
import trax
#from trax import layers as tl
#from trax.fastmath import numpy as fastnp
#from trax.supervised import training

# UNIT TEST for UNQ_C1
def test_get_conversation(target):

    data = {'file1.json': {'log':[{'text': 'hi'},
                                  {'text': 'hello'},
                                  {'text': 'nice'}]},
            'file2.json':{'log':[{'text': 'a b'}, 
                                 {'text': ''}, 
                                 {'text': 'good '}, 
                                 {'text': 'no?'}]}}
    
    res1 = target('file1.json', data)
    res2 = target('file2.json', data)
    
    expected1 = ' Person 1: hi Person 2: hello Person 1: nice'
    expected2 = ' Person 1: a b Person 2:  Person 1: good  Person 2: no?'

    success = 0
    fails = 0
    
    try:
        assert res1 == expected1
        success += 1
    except ValueError:
        print('Error in test 1 \nResult  : ', res1, 'x \nExpected: ', expected1)
        fails += 1
    try:
        assert res2 == expected2
        success += 1
    except:
        print('Error in test 2 \nResult  : ', res2, ' \nExpected: ', expected2)
        fails += 1
            
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")


# UNIT TEST for UNQ_C2
def test_reversible_layer_forward(target):
    f1 = lambda x: x + 2
    g1 = lambda x: x * 3
    
    f2 = lambda x: x + 1
    g2 = lambda x: x * 2
    
    input_vector1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    expected1 = np.array([8, 10, 12, 14, 29, 36, 43, 50])
    
    input_vector2 = np.array([1] * 128)
    expected2 = np.array([3] * 64 + [7] * 64)
    
    success = 0
    fails = 0
    try:
        res = target(input_vector1, f1, g1)
        assert isinstance(res, np.ndarray)
        success += 1
    except:
        print('Wrong type! Output is not of type np.ndarray')
        fails += 1
    try:
        res = target(input_vector1, f1, g1)
        assert np.allclose(res, expected1)
        success += 1
    except ValueError:
        print('Error in test 1 \nResult  : ', res, 'x \nExpected: ', expected1)
        fails += 1
    try:
        res = target(input_vector2, f2, g2)
        assert np.allclose(res, expected2)
        success += 1
    except:
        print('Error in test 2 \nResult  : ', res, ' \nExpected: ', expected2)
        fails += 1
            
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")


# UNIT TEST for UNQ_C3
def test_reversible_layer_reverse(target):
    
    f1 = lambda x: x + 2
    g1 = lambda x: x * 3
    
    f2 = lambda x: x + 1
    g2 = lambda x: x * 2
    
    input_vector1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    expected1 = np.array([-3,  0,  3,  6,  2,  0, -2, -4])
    
    input_vector2 = np.array([1] * 128)
    expected2 = np.array([1] * 64 + [-1] * 64)
    
    success = 0
    fails = 0
    try:
        res = target(input_vector1, f1, g1)
        assert isinstance(res, np.ndarray)
        success += 1
    except:
        print('Wrong type! Output is not of type np.ndarray')
        fails += 1
    try:
        res = target(input_vector1, f1, g1)
        assert np.allclose(res, expected1)
        success += 1
    except ValueError:
        print('Error in test 1 \nResult  : ', res, 'x \nExpected: ', expected1)
        fails += 1
    try:
        res = target(input_vector2, f2, g2)
        assert np.allclose(res, expected2)
        success += 1
    except:
        print('Error in test 2 \nResult  : ', res, ' \nExpected: ', expected2)
        fails += 1
            
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")
        

# UNIT TEST for UNQ_C4
def test_ReformerLM(target):
    test_cases = [
                {
                    "name":"layer_len_check",
                    "expected":11,
                    "error":"We found {} layers in your model. It should be 11.\nCheck the LSTM stack before the dense layer"
                },
                {
                    "name":"simple_test_check",
      "expected":"Serial[ShiftRight(1)Embedding_train_512DropoutPositionalEncodingDup_out2ReversibleSerial_in2_out2[ReversibleHalfResidualV2_in2_out2[Serial[LayerNorm]SelfAttention]ReversibleSwap_in2_out2ReversibleHalfResidualV2_in2_out2[Serial[LayerNormDense_2048DropoutFastGeluDense_512Dropout]]ReversibleSwap_in2_out2ReversibleHalfResidualV2_in2_out2[Serial[LayerNorm]SelfAttention]ReversibleSwap_in2_out2ReversibleHalfResidualV2_in2_out2[Serial[LayerNormDense_2048DropoutFastGeluDense_512Dropout]]ReversibleSwap_in2_out2]Concatenate_in2LayerNormDropoutDense_trainLogSoftmax]",
                    "error":"The ReformerLM is not defined properly."
                }
            ]
    temp_model = target('train')
    
    success = 0
    fails = 0
    
    for test_case in test_cases:
        try:
            if test_case['name'] == "simple_test_check":
                assert test_case["expected"] == str(temp_model).replace(' ', '').replace('\n','')
                success += 1
            if test_case['name'] == "layer_len_check":
                if test_case["expected"] == len(temp_model.sublayers):
                    success += 1
                else:
                    print(test_case["error"].format(len(temp_model.sublayers))) 
                    fails += 1
        except:
            print(test_case['error'])
            fails += 1
            
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")


# UNIT TEST for UNQ_C5
def test_tasks(train_task, eval_task):
    target = train_task
    success = 0
    fails = 0
     
    # Test the labeled data parameter for train_task
    try:
        strlabel = str(target._labeled_data)
        assert ("generator" in strlabel) and ("add_loss_weights" in  strlabel)
        success += 1
    except:
        fails += 1
        print("Wrong labeled data parameter in train_task")
    
    # Test the cross entropy loss data parameter
    try:
        strlabel = str(target._loss_layer)
        assert(strlabel == "CrossEntropyLoss_in3")
        success += 1
    except:
        fails += 1
        print("Wrong loss functions. CrossEntropyLoss_in3 was expected")
        
     # Test the optimizer parameter
    try:
        assert(isinstance(target.optimizer, trax.optimizers.adam.Adam))
        success += 1
    except:
        fails += 1
        print("Wrong optimizer")
        
    # Test the schedule parameter
    try:
        assert(isinstance(target._lr_schedule,trax.supervised.lr_schedules._BodyAndTail))
        success += 1
    except:
        fails += 1
        print("Wrong learning rate schedule type")
    
    # Test the _n_steps_per_checkpoint parameter
    try:
        assert(target._n_steps_per_checkpoint==10)
        success += 1
    except:
        fails += 1
        print("Wrong checkpoint step frequency")
        
    target = eval_task
    # Test the labeled data parameter for eval_task
    try:
        strlabel = str(target._labeled_data)
        assert ("generator" in strlabel) and ("add_loss_weights" in  strlabel)
        success += 1
    except:
        fails += 1
        print("Wrong labeled data parameter in eval_task")
    
    # Test the metrics in eval_task 
    try:
        strlabel = str(target._metrics).replace(' ', '')
        assert(strlabel == "[CrossEntropyLoss_in3,Accuracy_in3]")
        success += 1
    except:
        fails += 1
        print(f"Wrong metrics. found {strlabel} but expected [CrossEntropyLoss_in3,Accuracy_in3]")
        
        
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")
        

