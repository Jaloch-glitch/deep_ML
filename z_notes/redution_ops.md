# All Use the same axis parameter
<code>
    np.sum(X, axis = 1)
    np.max(X, axis = 0)
    np.min(X, axis = 1)
    np.mean(X, axis = 0)
    np.std(X, axis =1)
    np.var(X, axis =0)
</code>


**1. The axis confusion rule:**
axis = N means "collapse the dimention N"
- You coompute Along that axis
- result has that dimention removed
Logic : Axis 0, operations down coluumns and Axis 1, operarions accross rows 