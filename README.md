Company: Offering a variety of delectable and healthy cookies for every occasion, Kevin Cookie Company is a world leader in the cookie business.

about the dataset: The "Kevin Cookie Company Financials" dataset consisted of an excel spreadsheet saved in xlsx format. The dataset was connected to and loaded into Jupyter Notebook, and an in-depth analysis was conducted to better understand the nature of the company's activities and record structure. It was determined that the company has sales in five distinct countries across two continents: North America and Europe. The countries are: Canada, the United States, Mexico, France, and Germany. It was also revealed that the company only made six sorts of cookies: chocolate chip, white chocolate macadamia nut, oatmeal raisin, snickerdoodle, sugar, and fortune cookies.

Method Justification: The chosen data science methodologies are particularly appropriate for the dataset for numerous reasons:

Data cleaning and preprocessing tackle missing values and inconsistencies through meticulous cleaning. It identifies outliers utilizing boxplots and statistical summaries. StandardScaler is employed to standardize data, thereby ensuring stability for PCA and clustering.

Principal Component Analysis (PCA) diminishes dimensionality while preserving variance for enhanced financial insights. It retains components with eigenvalues exceeding 1, in accordance with the Kaiser Criterion.

K-Means clustering categorizes customers based on their financial performance.
