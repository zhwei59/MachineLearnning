import pandas as pd
ps=pd.read_csv('/Users/zhwei/share/lixianglan/gitrepo/kaggle/digitRecognizer/result.csv')
ps['ImageId']+=1
ps.to_csv('/Users/zhwei/share/lixianglan/gitrepo/kaggle/digitRecognizer/result2.csv', header=True,index=False)