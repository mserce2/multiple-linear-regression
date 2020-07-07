import pandas as pd
import numpy as np
from  sklearn.linear_model import LinearRegression

df=pd.read_csv("multipleLinear.csv",sep=";")
x=df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)

multiple_linear=LinearRegression()
multiple_linear.fit(x,y)

print("b0:",multiple_linear.intercept_)
print("b1,b2:",multiple_linear.coef_)

#yukarıda eksenleri kesen noktayı sonra eğimi bulduktan sonra;
#10 yıl çalışıp 35 yaşında olan bir kişinin maaşını predict ettik daha sonra;
#5 yıl çalışıp 35 yaşında olan birinin maaşını predict edip sonuçlara baktık
print(multiple_linear.predict(np.array([[10,35],[5,35]])))