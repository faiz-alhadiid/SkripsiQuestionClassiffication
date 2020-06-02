from sklearn.metrics import confusion_matrix, classification_report

def evaluate(actual, predicted, label):
    n = len(predicted)
    acc = sum(1 for a, b in zip(actual, predicted) if a==b)/n
    cr = classification_report(actual, predicted, label)
    cf = confusion_matrix(actual, predicted, label)
    precision = dict((x, 0) for x in label)
    recall = dict((x, 0) for x in label)
    f_measure = dict((x, 0) for x in label)
    m = len(label)
    for idx in label:
        
        precision[idx] = cr[idx]['precision']
        recall[idx] = cr[idx]['recall']
        f_measure[idx] = cr[idx]['f1-score']
    return {
        'conf_matrix': cf.tolist(),
        'precision': precision,
        'recall': recall,
        'f_measure': f_measure,
        'accuracy': cr['accuracy'],
        'avg_precision': cr['macro avg']['precision'],
        'avg_recall' : cr['macro avg']['recall'],
        'avg_f_measure': cr['macro avg']['f1-score']
    }