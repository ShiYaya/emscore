from scorer import EMScorer


if __name__ == '__main__':
    # you video's path list
    vids = ['emscore/example/Bfyp0C02sko_000248_000258.mp4']
    metric = EMScorer()
    # the candidate caption
    cands = ['A person is frying a pan of eggs that are colored pink and green.']
    refs = ['Two egg yolks are placed in food flavored white yolk on a pan until they are cooked.']
    # refs = ['Two egg yolks are placed in food flavored white yolk on a pan until they are cooked.']
    results = metric.score(cands=cands, refs=refs, vids=vids, idf=False)

    """
    only use video as groud truth
    """
    print('------------------------------------------------------------------------')
    print('fine-grained EMScore(X,V), P: {}, R:{}, F:{}'.format(
        results['EMScore(X,V)']['full_P'], results['EMScore(X,V)']['full_R'], results['EMScore(X,V)']['full_F']))
    print(
        'coarse-grained EMScore(X,V), {}'.format(results['EMScore(X,V)']['cogr']))
    print('fine- and coarse-grained EMScore(X,V), P: {}, R:{}, F:{}'.format(
        results['EMScore(X,V)']['full_P'], results['EMScore(X,V)']['full_R'], results['EMScore(X,V)']['full_F']))

    """
    only use references as groud truth 
    """
    print('------------------------------------------------------------------------')
    print('fine- and coarse-grained EMScore(X,X*), P: {}, R:{}, F:{}'.format(results['EMScore(X,X*)']['full_P'],
          results['EMScore(X,X*)']['full_R'], results['EMScore(X,X*)']['full_F']))

    """
    use both video and references as groud truth
    """
    print('------------------------------------------------------------------------')
    print('fine- and coarse-grained EMScore(X,V,X*), P: {}, R:{}, F:{}'.format(results['EMScore(X,V,X*)']['full_P'],
          results['EMScore(X,V,X*)']['full_R'], results['EMScore(X,V,X*)']['full_F']))
    print()
