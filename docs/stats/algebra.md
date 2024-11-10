# Algebra

## Calclus

### 1. Derivative (Single-Variable)
The **derivative** of a single-variable function, \( f(x) \), measures how \( f \) changes with respect to small changes in \( x \). For example, if \( f(x) = x^2 \), the derivative is:
\[
f'(x) = \frac{df}{dx} = 2x
\]
This derivative tells us the rate of change of \( f \) as \( x \) changes. In single-variable functions, we only have one input variable, so we only need one derivative to describe the function's slope at any given point.

#### Key Points
- The derivative applies to single-variable functions.
- It tells us how the function value changes as the single variable changes.
- It results in a single value representing the slope at a particular point.

---

### 2. Partial Derivatives (Multi-Variable Functions)
When dealing with functions of more than one variable, such as \( f(x, y) \), we use **partial derivatives**. A partial derivative represents how the function changes with respect to one variable while keeping the other variables constant.

For a function \( f(x, y) \), we can find:
- The partial derivative with respect to \( x \): \( \frac{\partial f}{\partial x} \)
- The partial derivative with respect to \( y \): \( \frac{\partial f}{\partial y} \)

These partial derivatives tell us the rate of change in the \( x \) and \( y \) directions, independently. This is useful in functions with multiple variables, where we might want to know the effect of changing one variable while holding others constant.

#### Example of Partial Derivatives
Consider the function:
\[
f(x, y) = x^2 + y^2
\]
The partial derivative of \( f \) with respect to \( x \) (treating \( y \) as constant) is:
\[
\frac{\partial f}{\partial x} = 2x
\]
The partial derivative of \( f \) with respect to \( y \) (treating \( x \) as constant) is:
\[
\frac{\partial f}{\partial y} = 2y
\]
These partial derivatives tell us:
- How \( f \) changes as \( x \) changes, with \( y \) held constant (\( 2x \) gives the rate of change in the \( x \)-direction).
- How \( f \) changes as \( y \) changes, with \( x \) held constant (\( 2y \) gives the rate of change in the \( y \)-direction).

---

### 3. Gradient (Vector of Partial Derivatives in Multi-Variable Functions)
The **gradient** is a vector that combines all partial derivatives of a multi-variable function. For a function \( f(x, y) \), the gradient \( \nabla f \) is:
\[
\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)
\]
This vector points in the direction of the steepest ascent of \( f \) and gives the rate of increase in that direction. 

For our function \( f(x, y) = x^2 + y^2 \), the gradient is:
\[
\nabla f = (2x, 2y)
\]
So, at any point \( (x, y) \), the gradient points in the direction of maximum increase of \( f(x, y) \), and its magnitude \( \| \nabla f \| \) gives the rate of that increase.

#### Key Points
- The gradient is a vector formed by the partial derivatives of a multi-variable function.
- It combines the information from each partial derivative, giving both the direction of steepest ascent and the rate of increase in that direction.
- In optimization, following the negative gradient direction leads to the steepest descent, making it useful for finding minima of functions.

---

### Summary Comparison: Derivative vs. Partial Derivative vs. Gradient
| **Concept**          | **Single-Variable Function**               | **Multi-Variable Function**                              |
|----------------------|--------------------------------------------|----------------------------------------------------------|
| **Derivative**       | Measures rate of change of \( f(x) \) as \( x \) changes. Single value representing slope. | Not directly applicable, as multi-variable functions require partial derivatives. |
| **Partial Derivative** | Not applicable in single-variable functions, since only one variable exists. | Measures rate of change of \( f(x, y, \dots) \) with respect to one variable, keeping others constant. |
| **Gradient**         | In single-variable, gradient is the derivative itself. Points in the direction of maximum increase of \( f(x) \) (positive or negative slope). | Vector of all partial derivatives. Points in the direction of the steepest ascent of \( f(x, y, \dots) \). Useful in optimization. |

In essence:
- **Derivatives** describe changes in single-variable functions.
- **Partial derivatives** describe changes in each direction in multi-variable functions.
- **Gradients** combine all partial derivatives into a vector, pointing in the direction of steepest ascent in multi-variable functions.

### References
- [Demystifying Eigenvalues and Eigenvectors: Understanding Linear Transformations and Data Analysis](https://towardsdev.com/demystifying-eigenvalues-and-eigenvectors-understanding-linear-transformations-and-data-analysis-82b86b3fd33b)
- [Derivatives in Data Science](https://imswapnilb.medium.com/derivatives-in-data-science-c5d7bd916f17)
