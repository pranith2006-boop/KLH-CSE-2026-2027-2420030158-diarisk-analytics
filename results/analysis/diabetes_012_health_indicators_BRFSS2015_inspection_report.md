# Dataset Inspection Report: diabetes_012_health_indicators_BRFSS2015.csv

## 1. Basic Metadata
- **Filename**: diabetes_012_health_indicators_BRFSS2015.csv
- **Number of Rows**: 253,680
- **Number of Columns**: 22
- **Number of Duplicate Rows**: 23,899 (9.42%)

## 2. Column Variables and Data Types
The dataset contains **3** numerical columns and **19** categorical columns (including binary flags and ordinal codes).

### Columns by Semantic Classification
- **Numerical (3):** BMI, MentHlth, PhysHlth
- **Categorical (19):** Diabetes_012, HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, DiffWalk, Sex, Age, Education, Income

### Complete Data Type Details
| Column Name | Pandas Dtype | Inferred Type |
| --- | --- | --- |
| `Diabetes_012` | `float64` | Categorical |
| `HighBP` | `float64` | Categorical |
| `HighChol` | `float64` | Categorical |
| `CholCheck` | `float64` | Categorical |
| `BMI` | `float64` | Numerical |
| `Smoker` | `float64` | Categorical |
| `Stroke` | `float64` | Categorical |
| `HeartDiseaseorAttack` | `float64` | Categorical |
| `PhysActivity` | `float64` | Categorical |
| `Fruits` | `float64` | Categorical |
| `Veggies` | `float64` | Categorical |
| `HvyAlcoholConsump` | `float64` | Categorical |
| `AnyHealthcare` | `float64` | Categorical |
| `NoDocbcCost` | `float64` | Categorical |
| `GenHlth` | `float64` | Categorical |
| `MentHlth` | `float64` | Numerical |
| `PhysHlth` | `float64` | Numerical |
| `DiffWalk` | `float64` | Categorical |
| `Sex` | `float64` | Categorical |
| `Age` | `float64` | Categorical |
| `Education` | `float64` | Categorical |
| `Income` | `float64` | Categorical |

## 3. Data Integrity & Completeness
### Missing Values
| Column Name | Missing Count | Percentage Missing |
| --- | --- | --- |
| `Diabetes_012` | 0 | 0.00% |
| `HighBP` | 0 | 0.00% |
| `HighChol` | 0 | 0.00% |
| `CholCheck` | 0 | 0.00% |
| `BMI` | 0 | 0.00% |
| `Smoker` | 0 | 0.00% |
| `Stroke` | 0 | 0.00% |
| `HeartDiseaseorAttack` | 0 | 0.00% |
| `PhysActivity` | 0 | 0.00% |
| `Fruits` | 0 | 0.00% |
| `Veggies` | 0 | 0.00% |
| `HvyAlcoholConsump` | 0 | 0.00% |
| `AnyHealthcare` | 0 | 0.00% |
| `NoDocbcCost` | 0 | 0.00% |
| `GenHlth` | 0 | 0.00% |
| `MentHlth` | 0 | 0.00% |
| `PhysHlth` | 0 | 0.00% |
| `DiffWalk` | 0 | 0.00% |
| `Sex` | 0 | 0.00% |
| `Age` | 0 | 0.00% |
| `Education` | 0 | 0.00% |
| `Income` | 0 | 0.00% |

> **Note**: No missing values were detected in any column.

### Suspicious and Invalid Values
- [WARNING] Column 'BMI' has 279 value(s) above BMI of 80.

## 4. Target Variables & Class Distributions
Identified target candidate(s): `Diabetes_012`

### Class Distribution for Target Column: `Diabetes_012`
| Class / Value | Count | Percentage |
| --- | --- | --- |
| 0.0 | 213,703 | 84.24% |
| 2.0 | 35,346 | 13.93% |
| 1.0 | 4,631 | 1.83% |

**Imbalance Analysis**: The class ratio between majority and minority is **46.15:1**. There is a **significant class imbalance** that may need to be addressed during modeling (e.g., SMOTE, class weighting).

## 5. Statistical Summaries
### Numerical Columns (Min/Max/Outliers)
| Column Name | Min Value | Max Value | Outliers (IQR Method) | Outlier Percentage |
| --- | --- | --- | --- | --- |
| `BMI` | 12.0 | 98.0 | 9,847 | 3.88% |
| `MentHlth` | 0.0 | 30.0 | 36,208 | 14.27% |
| `PhysHlth` | 0.0 | 30.0 | 40,949 | 16.14% |

### Categorical Columns Unique Values (Top 15 values displayed)
- **`Diabetes_012`** (3 unique values): [0.0, 2.0, 1.0]
- **`HighBP`** (2 unique values): [1.0, 0.0]
- **`HighChol`** (2 unique values): [1.0, 0.0]
- **`CholCheck`** (2 unique values): [1.0, 0.0]
- **`Smoker`** (2 unique values): [1.0, 0.0]
- **`Stroke`** (2 unique values): [0.0, 1.0]
- **`HeartDiseaseorAttack`** (2 unique values): [0.0, 1.0]
- **`PhysActivity`** (2 unique values): [0.0, 1.0]
- **`Fruits`** (2 unique values): [0.0, 1.0]
- **`Veggies`** (2 unique values): [1.0, 0.0]
- **`HvyAlcoholConsump`** (2 unique values): [0.0, 1.0]
- **`AnyHealthcare`** (2 unique values): [1.0, 0.0]
- **`NoDocbcCost`** (2 unique values): [0.0, 1.0]
- **`GenHlth`** (5 unique values): [5.0, 3.0, 2.0, 4.0, 1.0]
- **`DiffWalk`** (2 unique values): [1.0, 0.0]
- **`Sex`** (2 unique values): [0.0, 1.0]
- **`Age`** (13 unique values): [9.0, 7.0, 11.0, 10.0, 8.0, 13.0, 4.0, 6.0, 2.0, 12.0, 5.0, 1.0, 3.0]
- **`Education`** (6 unique values): [4.0, 6.0, 3.0, 5.0, 2.0, 1.0]
- **`Income`** (8 unique values): [3.0, 1.0, 8.0, 6.0, 4.0, 7.0, 2.0, 5.0]

### Basic Descriptives Table
```
Column                              | count      | mean       | std        | min        | 50%        | max       
--------------------------------------------------------------------------------------------------------------
Diabetes_012                        | 253680.0   | 0.30       | 0.70       | 0.00       | 0.00       | 2.00      
HighBP                              | 253680.0   | 0.43       | 0.49       | 0.00       | 0.00       | 1.00      
HighChol                            | 253680.0   | 0.42       | 0.49       | 0.00       | 0.00       | 1.00      
CholCheck                           | 253680.0   | 0.96       | 0.19       | 0.00       | 1.00       | 1.00      
BMI                                 | 253680.0   | 28.38      | 6.61       | 12.00      | 27.00      | 98.00     
Smoker                              | 253680.0   | 0.44       | 0.50       | 0.00       | 0.00       | 1.00      
Stroke                              | 253680.0   | 0.04       | 0.20       | 0.00       | 0.00       | 1.00      
HeartDiseaseorAttack                | 253680.0   | 0.09       | 0.29       | 0.00       | 0.00       | 1.00      
PhysActivity                        | 253680.0   | 0.76       | 0.43       | 0.00       | 1.00       | 1.00      
Fruits                              | 253680.0   | 0.63       | 0.48       | 0.00       | 1.00       | 1.00      
Veggies                             | 253680.0   | 0.81       | 0.39       | 0.00       | 1.00       | 1.00      
HvyAlcoholConsump                   | 253680.0   | 0.06       | 0.23       | 0.00       | 0.00       | 1.00      
AnyHealthcare                       | 253680.0   | 0.95       | 0.22       | 0.00       | 1.00       | 1.00      
NoDocbcCost                         | 253680.0   | 0.08       | 0.28       | 0.00       | 0.00       | 1.00      
GenHlth                             | 253680.0   | 2.51       | 1.07       | 1.00       | 2.00       | 5.00      
MentHlth                            | 253680.0   | 3.18       | 7.41       | 0.00       | 0.00       | 30.00     
PhysHlth                            | 253680.0   | 4.24       | 8.72       | 0.00       | 0.00       | 30.00     
DiffWalk                            | 253680.0   | 0.17       | 0.37       | 0.00       | 0.00       | 1.00      
Sex                                 | 253680.0   | 0.44       | 0.50       | 0.00       | 0.00       | 1.00      
Age                                 | 253680.0   | 8.03       | 3.05       | 1.00       | 8.00       | 13.00     
Education                           | 253680.0   | 5.05       | 0.99       | 1.00       | 5.00       | 6.00      
Income                              | 253680.0   | 6.05       | 2.07       | 1.00       | 7.00       | 8.00      
```
