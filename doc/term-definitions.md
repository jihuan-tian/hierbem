# Definition of terms

* \f$\mathcal{H}\f$-matrix node: In the current C++ implementation of \f$\mathcal{H}\f$-matrix, it is represented as a hierarchy of linked `HMatrix` objects, which are organized in a tree structure being the same as the associated block cluster tree (`BlockClusterTree`). Then an \f$\mathcal{H}\f$-matrix node is one of these `HMatrix` objects, which is a node in the tree.
* Minimum matrix dimension: Let \f$M \in \mathbb{R}^{m \times n}\f$, then the minimum matrix dimension is \f$\min\{m, n\}\f$. Then \f${\rm rank}(M) \leq \min\{m, n\}\f$. This term is frequently used in the implementation of `RkMatrix` and singular value decomposition (`LAPACKFullMatrixExt::svd` and `LAPACKFullMatrixExt::reduced_svd`) of a full matrix `LAPACKFullMatrixExt`.
* Global matrix: the matrix defined on the complete index set \f$I \times J\f$.
* Local matrix: the matrix defined on a block cluster \f$\tau \times \sigma\f$, which is a subset of the complete index set \f$I \times J\f$.
* Formal rank of a rank-k matrix: it is the number of columns in \f$A\f$ or \f$B\f$.
* Long matrix: the matrix has more number of rows than columns.
* Wide matrix: the matrix has more number of columns than rows.
