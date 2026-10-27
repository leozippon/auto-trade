# DPAL: A Test-Time Adaptation Loss for Financial Time Series Forecasting

- Authors: Wuzhou Lian, Haochun Wang, Yuefan Deng
- Venue: arXiv preprint
- arXiv: 2610.14376 (first version 2026-10-09; this text from v1, 2026-10-09)
- Licence: CC BY 4.0 (http://creativecommons.org/licenses/by/4.0/)
- Text here: arXiv HTML, full text without references

## Abstract

Test-time adaptation (TTA) enables time series forecasters to update after deployment as new observations become available. However, adaptation objectives that focus primarily on prediction errors at individual time steps do not fully address the needs of financial time series forecasting. In addition to pointwise return fitting, return direction and the price trajectories reconstructed from sequences of returns are also important. To better account for these properties, we introduce Direction- and Price-Aware Adaptation Loss (DPAL), which combines pointwise return fitting with directional and price-calibration terms. We apply DPAL within existing TSF-TTA frameworks without modifying the source forecaster or the adapter architecture. Across a diverse collection of financial datasets, DPAL reduces reconstructed-price mean squared error in most evaluated settings.

## Body

### 1 Introduction

Time series forecasting aims to predict future observations from historical temporal patterns. In financial markets, reliable forecasts support a wide range of decisions, including portfolio analysis, risk assessment, and capital allocation. Although modern forecasting methods ( Nie et al., 2023 ; Liu et al., 2024 ; Zeng et al., 2023 ) have achieved good performance on standard forecasting benchmarks, their deployment in financial markets remains challenging. In particular, asset returns evolve over time, leading to distribution shifts that may degrade predictive accuracy after deployment.

Test-time adaptation for time series forecasting (TSF-TTA) ( Kim et al., 2025 ; Medeiros et al., 2025 ; Grover and Etemad, 2025 ; Im and Kwon, 2026 ) provides a way to adjust a deployed forecaster as test targets become available. TAFAS ( Kim et al., 2025 ) keeps the source forecaster frozen and updates lightweight gated calibration modules using fully and partially observed targets. PETSA ( Medeiros et al., 2025 ) improves parameter efficiency through low-rank calibration modules while using a composite loss to preserve forecasting accuracy under limited adaptation capacity. Although these methods were developed for non-stationary time series forecasting, their original objectives do not explicitly account for two properties relevant to financial forecasting: whether a predicted return has the correct sign and whether return errors accumulate into a distorted price trajectory.

Mean squared error (MSE) measures numerical deviation but does not treat a sign reversal as a distinct type of forecasting error. For example, if the realized return is 1%, predictions of 3% and -1% have the same absolute error, although only the latter reverses the direction. This distinction matters whenever a decision uses the predicted sign of the return. In addition, pointwise return fitting does not directly constrain the price trajectory reconstructed by compounding a sequence of predicted returns. Small but persistent return biases can accumulate over the forecasting horizon and produce a growing discrepancy between predicted and realized price paths, as illustrated in Figure 1 .

Together, these considerations motivate an adaptation objective that jointly incorporates pointwise return fitting, penalties for predictions whose signs differ from the observed returns, and calibration of cumulative price trajectories. To this end, we introduce Direction- and Price-Aware Adaptation Loss (DPAL). Following Wang et al. (2026) , we evaluate DPAL under a matured-ground-truth protocol, in which the adapter is updated only after the complete target window becomes available, without using partially observed ground truth for adaptation. Specifically, a past forecast window is eligible for adaptation only when its entire target span has been observed before the current prediction begins. This temporal separation prevents adaptation supervision from overlapping with the evaluated prediction span and avoids using partially revealed values from the current window as adaptation targets. We apply the same DPAL objective to both TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) , while retaining their original adaptation modules. For TAFAS, DPAL augments the original MSE objective with directional and price-calibration terms. For PETSA, DPAL replaces the original Huber, frequency-domain, and patch-wise structural terms.

[Figure] Figure 1 : Illustration of constant upward bias and its impact on price reconstruction. (a) A small upward shift in the predicted return rate preserves the original temporal shape but induces directional misjudgments (red areas), which are not distinguished by standard MSE. (b) During price reconstruction, the effect of this persistent return shift accumulates through compounding, leading to a progressively widening deviation between the predicted and ground-truth price trajectories.

We evaluate TAFAS and PETSA equipped with DPAL on eight financial datasets assembled for this study, spanning diverse equity and broader market settings. The experiments use five source forecasters: PatchTST ( Nie et al., 2023 ) , iTransformer ( Liu et al., 2024 ) , DLinear ( Zeng et al., 2023 ) , OLS ( Toner and Darlow, 2024 ) , and FreTS ( Yi et al., 2023 ) . Each model is evaluated at forecasting horizons of 60, 90, and 120 trading days. Compared with the frozen source forecasters and the TAFAS and PETSA baselines, the DPAL variants reduce reconstructed-price MSE in the large majority of the evaluated settings.

The main contributions of this work are summarized as follows:

- • We highlight that existing TSF-TTA objectives do not explicitly account for two properties relevant to financial time series forecasting: return direction and compounded price trajectories.

- • We introduce DPAL, which combines MSE, a directional loss, and a price-calibration loss, and we instantiate it in TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) .

- • We evaluate TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) equipped with DPAL across eight financial datasets, five source forecasters, and three forecasting horizons. The results show lower reconstructed-price MSE in the large majority of the evaluated settings.

### 2 Related Work

#### 2.1 Deep Learning in Time Series Forecasting

Recent advances in long-term time series forecasting (LTSF) include Transformer-based models such as PatchTST ( Nie et al., 2023 ) and iTransformer ( Liu et al., 2024 ) , linear models such as DLinear ( Zeng et al., 2023 ) and OLS ( Toner and Darlow, 2024 ) , and frequency-domain methods such as FreTS ( Yi et al., 2023 ) . These models are commonly evaluated on datasets from domains such as energy ( Maleki et al., 2024 ) , traffic ( Yin et al., 2022 ; Zhang et al., 2024 ) , and weather ( Wu et al., 2023 ; Verma et al., 2024 ) . These approaches have demonstrated competitive forecasting performance on standard LTSF benchmarks.

#### 2.2 Test-Time Adaptation for Distribution Shifts

As data distributions shift over time, time series forecasters may experience performance degradation after deployment. In this setting, test-time adaptation (TTA) ( Kim et al., 2025 ; Medeiros et al., 2025 ; Grover and Etemad, 2025 ; Im and Kwon, 2026 ) has been introduced into time series forecasting, using information that becomes available after deployment to update a pretrained forecaster. TAFAS ( Kim et al., 2025 ) freezes the source forecaster and updates gated calibration modules (GCMs) using fully and partially observed targets, with an adaptation schedule based on temporal periodicity. PETSA ( Medeiros et al., 2025 ) reduces the trainable parameter count through low-rank input and output calibration modules and optimizes a composite objective containing robust, frequency-domain, and patch-wise structural terms. DynaTTA ( Grover and Etemad, 2025 ) estimates distribution shift from prediction errors and representation drift, then adjusts the adaptation rate and a shift-conditioned gate that controls the influence of its calibration modules. COSA ( Im and Kwon, 2026 ) performs gated residual correction in the output space using context from recently observed targets. Despite their differences in architecture and adaptation strategy, their adaptation objectives are not specifically designed for financial time series forecasting.

#### 2.3 Objectives and Forecasting in Financial Time Series

Deep stock forecasting models commonly use mean squared error (MSE) either as the primary objective or together with auxiliary objectives. MASTER ( Li et al., 2024 ) uses market-guided feature selection and Transformer-based aggregation to predict normalized stock returns with MSE, whereas FactorGCL ( Duan et al., 2025 ) combines recurrent and hypergraph encoders and jointly optimizes multi-horizon return MSE and a temporal residual contrastive loss. To better align prediction with stock selection, RSR ( Feng et al., 2019 ) , STHAN-SR ( Sawhney et al., 2021 ) , and StockMixer ( Fan and Shen, 2024 ) augment pointwise squared-error regression with pairwise ranking losses, thereby preserving return-magnitude supervision while encouraging correct cross-stock ordering. Their evaluations further consider ranking quality and portfolio performance. Directional information has also been incorporated into forecasting objectives. MADL ( Michańków et al., 2024 ) uses discrete sign agreement between predicted and realized returns, weighted by realized return magnitudes, for trading signal generation. CosDir ( Lee et al., 2026 ) augments MSE with cosine alignment between the first-difference vectors of predictions and targets. DPAL uses a pointwise rectified-product penalty on original-scale returns and combines it with pointwise return fitting and compounded price-path calibration for test-time adaptation.

### 3 DPAL: Direction- and Price-Aware Adaptation Loss

Recall that existing TSF-TTA objectives do not explicitly account for return direction or compounded price trajectories in financial time series forecasting. To incorporate these properties into the adaptation objective, we introduce Direction- and Price-Aware Adaptation Loss (DPAL), which changes only the adaptation objective while leaving the forecasting backbone and the test-time adapter unchanged.

To formulate the loss components, let $b\in\{1,\ldots,B\}$ denote the batch index, $h\in\{1,\ldots,H\}$ the forecast position, and $c\in\{1,\ldots,C\}$ the target-series index. The pointwise mean squared error is computed in the normalized return space. For the directional and price-aware terms, the predicted and ground-truth returns are transformed back to the original return scale using the inverse normalization transform defined for each dataset. We denote the corresponding original-scale returns by $\hat{r}_{b,h,c}$ and $r_{b,h,c}$ , respectively.

#### 3.1 Direction-Aware Loss

When pointwise MSE is used as the test-time adaptation objective, the expected squared error is minimized by the conditional mean of the target return given the available inputs. In financial return forecasting, much of the observed variation may be difficult to predict from the available inputs. In such settings, MSE-based adaptation may produce predictions that remain near the center of the return distribution and vary less than the observed returns. Figure 2 provides an illustrative example: the adapted predictions are concentrated within a narrow range of small positive values, whereas the observed returns vary more widely and frequently change sign. Such predictions may achieve relatively small pointwise error while failing to reflect changes in return direction.

[Figure] Figure 2 : Illustrative example of MSE-based test-time adaptation on AAPL. The figure shows return predictions from iTransformer with TAFAS without POGT, using both a look-back window and a forecasting horizon of 60 trading days. The vertical dashed line marks the beginning of the forecasting window. In this example, the adapted predictions remain within a narrow range of small positive values while the observed returns frequently change sign.

To incorporate return direction into the adaptation objective, we add a directional loss that explicitly penalizes predictions whose signs oppose the observed returns. Figure 3 provides an overview.

[Figure] Figure 3 : Overview of Direction-Aware Loss.

We construct the directional loss $\mathcal{L}_{\mathrm{dir}}$ by taking the negative product of the predicted and observed returns, followed by a ReLU activation function. For batch index $b\in\{1,\ldots,B\}$ , forecast position $h\in\{1,\ldots,H\}$ , and target series $c\in\{1,\ldots,C\}$ , the directional loss is

$$\mathcal{L}_{\mathrm{dir}}=\frac{1}{BHC}\sum_{b=1}^{B}\sum_{h=1}^{H}\sum_{c=1}^{C}\text{ReLU}(-\hat{r}_{b,h,c}\cdot r_{b,h,c}).$$ (1)

When predicted and observed returns have matching signs, $\hat{r}_{b,h,c}\cdot r_{b,h,c}>0$ . The negative product passed to ReLU is below zero, so the corresponding loss term is zero, yielding no penalty. For returns with opposing signs, $\hat{r}_{b,h,c}\cdot r_{b,h,c}<0$ , and ReLU returns the positive penalty $\lvert\hat{r}_{b,h,c}\cdot r_{b,h,c}\rvert$ , whose magnitude depends on both returns.

#### 3.2 Price Calibration Loss

The directional loss penalizes opposite-sign predictions at individual forecast steps, but it does not account for how return errors accumulate over the forecasting horizon. This cumulative effect can be represented through the price trajectory reconstructed by compounding the sequence of returns. To account for this effect, we introduce a price calibration loss to reduce discrepancies between the trajectories reconstructed from predicted and observed returns. Both trajectories are initialized at 1, allowing the loss to measure discrepancies in compounded returns without depending on the nominal price of an asset. Figure 4 provides an overview.

[Figure] Figure 4 : Overview of Price Calibration Loss.

By setting the initial relative price to 1 and compounding the predicted and observed returns in the original scale, we define the predicted and true price trajectories at step $t\in\{1,\ldots,H\}$ , respectively, as

$$\hat{p}_{b,t,c}=\prod_{h=1}^{t}(1+\hat{r}_{b,h,c}),\quad p_{b,t,c}=\prod_{h=1}^{t}(1+r_{b,h,c}).$$ (2)

##### 3.2.1 Early Forecast Step Selection

Because prices are reconstructed by compounding predicted returns, each reconstructed price at a later forecast step depends on a longer sequence of return predictions. As a result, return errors can accumulate over the forecast horizon and lead to trajectory drift. To limit the influence of discrepancies arising after many compounding steps, we apply the price calibration loss only to the first $k$ forecasting steps. We denote the corresponding truncated trajectories by $\hat{p}^{1:k}_{b,t,c}$ and $p^{1:k}_{b,t,c}$ . In the evaluated implementation, $k=15$ for $H=60$ , and $k=30$ for $H\in\{90,120\}$ . The candidate combinations and their comparative results are reported in Appendix A.4 .

##### 3.2.2 Relative Price Trajectory Discrepancy

To compare the predicted and observed reconstructed trajectories without directly depending on their absolute levels, we measure their discrepancy in relative terms. We define the relative price ratio as

$$q_{b,t,c}=\frac{\hat{p}^{1:k}_{b,t,c}}{p^{1:k}_{b,t,c}}.$$ (3)

A value of $q_{b,t,c}=1$ indicates exact agreement between the two reconstructed prices, while deviations from 1 represent multiplicative discrepancies relative to the observed trajectory.

Compounding can occasionally produce large residuals in the relative price ratio, particularly when return deviations accumulate over multiple forecast steps. We therefore apply Huber loss, with the threshold parameter set to $\delta=1.0$ , to the residual $q_{b,t,c}-1$ . The quadratic region provides smooth sensitivity near the target, while the linear region limits the influence of large residuals.

For batch index $b\in\{1,\ldots,B\}$ , forecast position $t\in\{1,\ldots,k\}$ , and target series $c\in\{1,\ldots,C\}$ , the price calibration loss is

$$\mathcal{L}_{\mathrm{price}}=\frac{1}{BkC}\sum_{b=1}^{B}\sum_{t=1}^{k}\sum_{c=1}^{C}\text{Huber}\left(q_{b,t,c},1\right).$$ (4)

#### 3.3 Combined Adaptation Objective

DPAL combines pointwise return fitting in normalized space with directional and price constraints computed from returns in the original scale:

$$\mathcal{L}_{\mathrm{DPAL}}=\mathcal{L}_{\mathrm{MSE}}+\alpha\mathcal{L}_{\mathrm{dir}}+\beta\mathcal{L}_{\mathrm{price}}.$$ (5)

Here, $\alpha$ and $\beta$ control the contributions of the directional loss and price calibration loss, respectively. Because the three loss components differ in numerical scale, we use $\alpha=10^{5}$ and $\beta=10^{2}$ as a common weighting configuration throughout the main experiments. The same coefficients are used across all datasets, forecasting backbones, prediction horizons, and adaptation frameworks, without selecting coefficients separately for each setting. Section 4.3 examines the robustness of DPAL to alternative coefficient values.

### 4 Experiments

#### 4.1 Experimental Setup

##### 4.1.1 Datasets

We construct eight financial datasets spanning a range of equity and broader market settings. Seven are manually curated equity panels, while the remaining dataset contains market-level series from several asset classes. Complete ticker lists, preprocessing procedures, trading-date alignment, and chronological splits are provided in Appendix B .

- A. Magnificent 7: Seven widely followed U.S. mega-cap companies with substantial exposure to technology and technology-driven businesses.

- B. S&P 500 Excerpt: Twenty constituents selected from the S&P 500, covering a broader cross-section of established large-cap U.S. companies.

- C. Russell 2000 Excerpt: Twenty constituents selected from the Russell 2000 to represent the smaller-cap segment of the U.S. equity market.

- D. Technology Sector: Twenty technology-oriented companies spanning semiconductors, software, hardware, networking infrastructure, and digital services.

- E. Consumer Sector: Twenty companies from consumer-related industries, including consumer staples, retail, apparel, home improvement, and restaurants.

- F. Energy Sector: Twenty companies covering multiple parts of the energy industry, including integrated oil and gas, exploration and production, oilfield services, refining, and midstream operations.

- G. Nasdaq Mixed Excerpt: Twenty manually selected Nasdaq-listed companies from multiple industries, including health care, industrials, technology, communication services, consumer services, and utilities.

- H. Macro and Market Series: Ten market-level series comprising six major U.S. equity indices, one Treasury yield index, one currency index, and two commodity futures series.

For all datasets, each constituent series is treated as an individual prediction target, with no cross-series aggregation or weighting. The targets are daily simple returns computed from adjusted closing values.

##### 4.1.2 Forecasting Backbones and Comparison Settings

We use five forecasting backbones: PatchTST ( Nie et al., 2023 ) , iTransformer ( Liu et al., 2024 ) , DLinear ( Zeng et al., 2023 ) , OLS ( Toner and Darlow, 2024 ) , and FreTS ( Yi et al., 2023 ) . This collection includes Transformer-based, linear, and frequency-domain forecasting models and allows us to examine DPAL across different forecasting architectures.

For each backbone, we evaluate five settings. Base denotes the frozen pretrained source forecaster without test-time adaptation. TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) denote the corresponding test-time adaptation frameworks. In our evaluation, their updates use only completed forecast windows, and partially observed targets are not used for gradient updates. The remaining settings are TAFAS with DPAL and PETSA with DPAL . In both settings, DPAL is used as the adaptation objective while the corresponding adapter and source forecaster are retained.

##### 4.1.3 Implementation Details

The historical input length is set to $L=60$ trading days, and the forecast horizon is $H\in\{60,90,120\}$ trading days. All five backbones use dataset-level standard scaling. TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) retain their original mechanisms for constructing adaptation batches from detected temporal periods, along with their framework-specific configurations.

##### 4.1.4 Evaluation Protocol

The forecasting and evaluation pipeline uses two separate transformations. Dataset scale standardization is used during model estimation and history-conditioned rescaling is used during final evaluation. These transformations provide a fixed standardization scale for model estimation and incorporate recent return levels and variability into the mapping of predictions for evaluation, respectively.

Let $w\in\{1,\dots,W\}$ index rolling forecast windows, $i\in\{1,\dots,L\}$ index positions in the historical input, and $c\in\{1,\dots,C\}$ index target series. For each target series $c$ , $\mu^{\mathrm{sc}}_{c}$ and $\sigma^{\mathrm{sc}}_{c}$ denote the mean and standard deviation, respectively, estimated exclusively from the training portion of the dataset. The historical input $r^{\mathrm{hist}}_{w,i,c}$ is transformed by the standard scaler

$$r^{\mathrm{norm}}_{w,i,c}=\frac{r^{\mathrm{hist}}_{w,i,c}-\mu^{\mathrm{sc}}_{c}}{\sigma^{\mathrm{sc}}_{c}}.$$ (6)

The standardized history $r^{\mathrm{norm}}_{w,i,c}$ is then passed to the forecasting model to obtain the standardized return forecast $\hat{r}^{\mathrm{norm}}_{w,h,c}$ , where $h\in\{1,\dots,H\}$ . During adaptation, the normalized predicted and ground-truth returns are transformed back to the original scale using the scaler statistics $\mu^{\mathrm{sc}}_{c}$ and $\sigma^{\mathrm{sc}}_{c}$ to compute the direction-aware and price calibration losses. This ensures that the return signs and compounding operations in these losses are evaluated in the original return units rather than standardized values.

For final evaluation, we locally rescale the predicted returns with the aim of aligning their central level and variability with recent market conditions. We calculate the mean and standard deviation directly from the raw historical return window associated with each forecast window:

$$\mu^{\mathrm{hist}}_{w,c}=\frac{1}{L}\sum_{i=1}^{L}r^{\mathrm{hist}}_{w,i,c},\quad\sigma^{\mathrm{hist}}_{w,c}=\operatorname{Std}\!\left(r^{\mathrm{hist}}_{w,1:L,c}\right).$$ (7)

The standardized prediction is locally rescaled for evaluation as

$$\hat{r}^{\mathrm{real}}_{w,h,c}=\hat{r}^{\mathrm{norm}}_{w,h,c}\sigma^{\mathrm{hist}}_{w,c}+\mu^{\mathrm{hist}}_{w,c}.$$ (8)

Equation 8 rescales the standardized predictions using the mean and standard deviation computed from the observed look-back window and held fixed across the forecast horizon. The same rescaling rule is applied to the predictions of all compared methods, while the ground-truth returns used for evaluation are taken directly from the original return data.

We evaluate reconstructed relative price trajectories and set their value at each forecast origin to 1 for scale-invariant evaluation across assets and rolling windows. Let $\hat{r}^{\mathrm{real}}_{w,h,c}$ and $r^{\mathrm{real}}_{w,h,c}$ denote the predicted and observed future returns on the original return scale, respectively. For a forecast window, the predicted and observed relative prices at step $t$ are

$$\hat{p}_{w,t,c}=\prod_{h=1}^{t}\left(1+\hat{r}^{\mathrm{real}}_{w,h,c}\right),\qquad p_{w,t,c}=\prod_{h=1}^{t}\left(1+r^{\mathrm{real}}_{w,h,c}\right),\qquad t\in\{1,\dots,H\}.$$ (9)

For window index $w\in\{1,\dots,W\}$ , forecast position $t\in\{1,\dots,H\}$ , and target series $c\in\{1,\dots,C\}$ , we compute the price MSE over the complete forecast horizon:

$$\text{MSE}_{\mathrm{price}}=\frac{1}{WHC}\sum_{w=1}^{W}\sum_{t=1}^{H}\sum_{c=1}^{C}(\hat{p}_{w,t,c}-p_{w,t,c})^{2}.$$ (10)

#### 4.2 Main Results

##### 4.2.1 Forecasting Performance

Tables 1 and 2 report the MSE of reconstructed relative prices for experiments based on TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) , respectively. Lower values indicate better forecasting performance.

Using DPAL as the adaptation objective reduces reconstructed-price MSE in most configurations for both TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) across the evaluated datasets and all three forecast horizons. Each of the five forecasting backbones exhibits reductions in most configurations, suggesting that DPAL is not tied to a single adaptation mechanism. An ablation study evaluating the individual contributions of the direction-aware loss and the price calibration loss, as well as all possible combinations of the loss components, is provided in Appendix A.1 .

Additional quantitative results are provided in Appendix A.2 . Specifically, Appendix A.2.1 reports the relative improvements in reconstructed-price MSE across datasets, backbones, and forecast horizons, while Appendix A.2.2 presents supplementary MSE and MAE results computed directly in the standardized return space before inverse transformation. These results provide complementary views of the magnitude of the improvements in the primary evaluation metric and the pointwise accuracy of the predicted returns. As a robustness check, Appendix A.2.3 additionally evaluates reconstructed-price MSE using the same training-set inverse standardization used to compute DPAL’s directional and price calibration losses during adaptation.

Table: Table 1 : Forecasting performance comparison of TAFAS. Lower MSE is better. The best result within each backbone and horizon is highlighted in green.

| Datasets | $H$ | iTransformer | DLinear | FreTS | PatchTST | OLS |
| Base | TAFAS | with DPAL | Base | TAFAS | with DPAL | Base | TAFAS | with DPAL | Base | TAFAS | with DPAL | Base | TAFAS | with DPAL |
| Magnificent 7 | 60 | 1.617 | 1.137 | 0.547 | 0.754 | 0.742 | 0.477 | 0.530 | 0.528 | 0.476 | 1.502 | 0.764 | 0.489 | 1.784 | 1.118 | 0.515 |
| 90 | 6.164 | 3.634 | 1.482 | 2.062 | 1.993 | 0.943 | 1.152 | 1.152 | 0.933 | 5.265 | 2.293 | 1.139 | 5.761 | 2.864 | 1.379 |
| 120 | 17.717 | 9.087 | 2.780 | 3.879 | 3.685 | 1.503 | 2.030 | 2.032 | 1.616 | 13.147 | 5.506 | 2.075 | 15.585 | 6.792 | 3.004 |
| S&P 500 Excerpt | 60 | 0.310 | 0.236 | 0.104 | 0.127 | 0.126 | 0.086 | 0.080 | 0.079 | 0.075 | 0.287 | 0.138 | 0.094 | 0.360 | 0.239 | 0.115 |
| 90 | 1.084 | 0.758 | 0.286 | 0.307 | 0.293 | 0.164 | 0.153 | 0.152 | 0.137 | 0.794 | 0.319 | 0.197 | 1.187 | 0.717 | 0.319 |
| 120 | 3.662 | 2.501 | 0.889 | 0.613 | 0.594 | 0.289 | 0.275 | 0.273 | 0.231 | 2.415 | 0.735 | 0.499 | 4.108 | 2.485 | 0.979 |
| Russell 2000 Excerpt | 60 | 0.305 | 0.223 | 0.105 | 0.123 | 0.126 | 0.073 | 0.068 | 0.069 | 0.066 | 0.237 | 0.100 | 0.071 | 0.310 | 0.149 | 0.081 |
| 90 | 0.699 | 0.408 | 0.153 | 0.296 | 0.287 | 0.143 | 0.143 | 0.143 | 0.127 | 0.622 | 0.231 | 0.135 | 0.795 | 0.352 | 0.150 |
| 120 | 1.454 | 0.919 | 0.339 | 0.503 | 0.493 | 0.262 | 0.246 | 0.246 | 0.226 | 1.198 | 0.535 | 0.279 | 1.586 | 0.834 | 0.331 |
| Technology Sector | 60 | 0.422 | 0.214 | 0.102 | 0.162 | 0.153 | 0.097 | 0.097 | 0.098 | 0.089 | 0.307 | 0.115 | 0.095 | 0.466 | 0.196 | 0.102 |
| 90 | 1.263 | 0.515 | 0.187 | 0.352 | 0.338 | 0.172 | 0.184 | 0.186 | 0.159 | 0.838 | 0.255 | 0.172 | 1.265 | 0.397 | 0.197 |
| 120 | 2.852 | 1.316 | 0.313 | 0.598 | 0.584 | 0.277 | 0.304 | 0.304 | 0.255 | 1.779 | 0.547 | 0.277 | 2.857 | 1.022 | 0.324 |
| Consumer Sector | 60 | 0.026 | 0.023 | 0.013 | 0.014 | 0.014 | 0.011 | 0.010 | 0.010 | 0.010 | 0.021 | 0.016 | 0.012 | 0.026 | 0.022 | 0.013 |
| 90 | 0.057 | 0.051 | 0.026 | 0.028 | 0.028 | 0.021 | 0.018 | 0.018 | 0.017 | 0.044 | 0.032 | 0.024 | 0.059 | 0.049 | 0.027 |
| 120 | 0.113 | 0.102 | 0.049 | 0.046 | 0.046 | 0.034 | 0.029 | 0.030 | 0.028 | 0.081 | 0.058 | 0.043 | 0.112 | 0.094 | 0.051 |
| Energy Sector | 60 | 0.037 | 0.036 | 0.024 | 0.027 | 0.027 | 0.023 | 0.019 | 0.019 | 0.019 | 0.036 | 0.030 | 0.023 | 0.039 | 0.035 | 0.024 |
| 90 | 0.082 | 0.079 | 0.048 | 0.055 | 0.054 | 0.044 | 0.035 | 0.035 | 0.036 | 0.075 | 0.060 | 0.046 | 0.085 | 0.076 | 0.049 |
| 120 | 0.150 | 0.146 | 0.086 | 0.093 | 0.093 | 0.073 | 0.059 | 0.059 | 0.060 | 0.143 | 0.110 | 0.085 | 0.158 | 0.143 | 0.091 |
| Nasdaq Mixed Excerpt | 60 | 0.058 | 0.051 | 0.031 | 0.035 | 0.035 | 0.028 | 0.026 | 0.026 | 0.026 | 0.052 | 0.036 | 0.029 | 0.059 | 0.045 | 0.032 |
| 90 | 0.125 | 0.110 | 0.059 | 0.071 | 0.070 | 0.051 | 0.049 | 0.049 | 0.044 | 0.109 | 0.071 | 0.053 | 0.128 | 0.094 | 0.063 |
| 120 | 0.223 | 0.197 | 0.104 | 0.113 | 0.112 | 0.079 | 0.077 | 0.077 | 0.068 | 0.192 | 0.124 | 0.090 | 0.228 | 0.169 | 0.107 |
| Macro and Market Series | 60 | 0.019 | 0.016 | 0.011 | 0.012 | 0.012 | 0.011 | 0.009 | 0.009 | 0.009 | 0.018 | 0.012 | 0.010 | 0.020 | 0.015 | 0.011 |
| 90 | 0.040 | 0.035 | 0.021 | 0.026 | 0.025 | 0.020 | 0.017 | 0.017 | 0.017 | 0.037 | 0.024 | 0.019 | 0.045 | 0.033 | 0.022 |
| 120 | 0.073 | 0.061 | 0.036 | 0.040 | 0.040 | 0.031 | 0.026 | 0.026 | 0.026 | 0.061 | 0.040 | 0.031 | 0.079 | 0.057 | 0.035 |

Table: Table 2 : Forecasting performance comparison of PETSA. Lower MSE is better. The best result within each backbone and horizon is highlighted in green.

| Datasets | $H$ | iTransformer | DLinear | FreTS | PatchTST | OLS |
| Base | PETSA | with DPAL | Base | PETSA | with DPAL | Base | PETSA | with DPAL | Base | PETSA | with DPAL | Base | PETSA | with DPAL |
| Magnificent 7 | 60 | 1.617 | 1.079 | 0.782 | 0.754 | 0.765 | 0.502 | 0.530 | 0.578 | 0.469 | 1.502 | 0.709 | 0.489 | 1.784 | 1.107 | 0.646 |
| 90 | 6.164 | 3.459 | 1.847 | 2.062 | 1.826 | 1.013 | 1.152 | 1.199 | 0.888 | 5.265 | 1.968 | 1.106 | 5.761 | 2.454 | 1.299 |
| 120 | 17.717 | 7.588 | 3.900 | 3.879 | 3.603 | 1.552 | 2.030 | 2.142 | 1.463 | 13.147 | 5.283 | 2.654 | 15.585 | 5.352 | 2.783 |
| S&P 500 Excerpt | 60 | 0.310 | 0.172 | 0.122 | 0.127 | 0.144 | 0.084 | 0.080 | 0.087 | 0.075 | 0.287 | 0.108 | 0.103 | 0.360 | 0.180 | 0.125 |
| 90 | 1.084 | 0.434 | 0.320 | 0.307 | 0.322 | 0.164 | 0.153 | 0.164 | 0.135 | 0.794 | 0.211 | 0.200 | 1.187 | 0.430 | 0.333 |
| 120 | 3.662 | 1.322 | 0.963 | 0.613 | 0.681 | 0.289 | 0.275 | 0.302 | 0.225 | 2.415 | 0.553 | 0.470 | 4.108 | 1.268 | 0.947 |
| Russell 2000 Excerpt | 60 | 0.305 | 0.192 | 0.118 | 0.123 | 0.138 | 0.075 | 0.068 | 0.081 | 0.065 | 0.237 | 0.106 | 0.084 | 0.310 | 0.142 | 0.083 |
| 90 | 0.699 | 0.610 | 0.151 | 0.296 | 0.241 | 0.137 | 0.143 | 0.168 | 0.127 | 0.622 | 0.218 | 0.147 | 0.795 | 0.287 | 0.160 |
| 120 | 1.454 | 0.767 | 0.365 | 0.503 | 0.425 | 0.251 | 0.246 | 0.289 | 0.225 | 1.198 | 0.450 | 0.298 | 1.586 | 0.632 | 0.331 |
| Technology Sector | 60 | 0.422 | 0.196 | 0.099 | 0.162 | 0.214 | 0.101 | 0.097 | 0.116 | 0.091 | 0.307 | 0.141 | 0.103 | 0.466 | 0.201 | 0.100 |
| 90 | 1.263 | 0.459 | 0.193 | 0.352 | 0.290 | 0.178 | 0.184 | 0.226 | 0.160 | 0.838 | 0.285 | 0.185 | 1.265 | 0.454 | 0.179 |
| 120 | 2.852 | 0.937 | 0.341 | 0.598 | 0.660 | 0.293 | 0.304 | 0.400 | 0.249 | 1.779 | 0.433 | 0.327 | 2.857 | 0.948 | 0.313 |
| Consumer Sector | 60 | 0.026 | 0.020 | 0.013 | 0.014 | 0.016 | 0.011 | 0.010 | 0.012 | 0.010 | 0.021 | 0.016 | 0.012 | 0.026 | 0.021 | 0.013 |
| 90 | 0.057 | 0.045 | 0.026 | 0.028 | 0.031 | 0.021 | 0.018 | 0.022 | 0.017 | 0.044 | 0.031 | 0.022 | 0.059 | 0.043 | 0.026 |
| 120 | 0.113 | 0.087 | 0.047 | 0.046 | 0.049 | 0.033 | 0.029 | 0.035 | 0.026 | 0.081 | 0.054 | 0.038 | 0.112 | 0.078 | 0.046 |
| Energy Sector | 60 | 0.037 | 0.033 | 0.025 | 0.027 | 0.028 | 0.023 | 0.019 | 0.020 | 0.019 | 0.036 | 0.029 | 0.027 | 0.039 | 0.033 | 0.027 |
| 90 | 0.082 | 0.069 | 0.053 | 0.055 | 0.058 | 0.043 | 0.035 | 0.038 | 0.036 | 0.075 | 0.056 | 0.052 | 0.085 | 0.071 | 0.055 |
| 120 | 0.150 | 0.127 | 0.094 | 0.093 | 0.100 | 0.075 | 0.059 | 0.065 | 0.059 | 0.143 | 0.100 | 0.089 | 0.158 | 0.130 | 0.096 |
| Nasdaq Mixed Excerpt | 60 | 0.058 | 0.041 | 0.031 | 0.035 | 0.036 | 0.029 | 0.026 | 0.028 | 0.026 | 0.052 | 0.034 | 0.029 | 0.059 | 0.044 | 0.032 |
| 90 | 0.125 | 0.093 | 0.058 | 0.071 | 0.069 | 0.050 | 0.049 | 0.053 | 0.044 | 0.109 | 0.067 | 0.056 | 0.128 | 0.086 | 0.061 |
| 120 | 0.223 | 0.165 | 0.100 | 0.113 | 0.110 | 0.078 | 0.077 | 0.085 | 0.067 | 0.192 | 0.116 | 0.088 | 0.228 | 0.149 | 0.101 |
| Macro and Market Series | 60 | 0.019 | 0.016 | 0.012 | 0.012 | 0.014 | 0.011 | 0.009 | 0.009 | 0.009 | 0.018 | 0.012 | 0.011 | 0.020 | 0.017 | 0.011 |
| 90 | 0.040 | 0.029 | 0.020 | 0.026 | 0.025 | 0.020 | 0.017 | 0.018 | 0.017 | 0.037 | 0.024 | 0.021 | 0.045 | 0.031 | 0.021 |
| 120 | 0.073 | 0.048 | 0.036 | 0.040 | 0.041 | 0.031 | 0.026 | 0.027 | 0.026 | 0.061 | 0.042 | 0.034 | 0.079 | 0.052 | 0.035 |

##### 4.2.2 Rolling Price Trajectories

In addition to the window-level reconstructed-price MSE, we examine rolling price trajectories to show how return prediction errors accumulate across successive forecast blocks. At each forecast origin, the model predicts returns for the next $H$ trading days, after which the forecast origin is advanced by $H$ trading days to the next non-overlapping block. Unlike the primary evaluation metric, which resets the relative price to 1 at each forecast origin, this visualization initializes the relative price only once and then compounds predicted returns across consecutive blocks. This provides a complementary view of cumulative trajectory drift over the test period.

Figure 5 presents eight representative cases, with one case drawn from each dataset. The cases are selected to cover both adaptation frameworks and a range of forecasting backbones and horizons. For each dataset, we use the first series in the stored ordering as a fixed selection rule. Collectively, the eight cases cover all five forecasting backbones, all three forecasting horizons, $H\in\{60,90,120\}$ , and both TTA frameworks, TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) .

[Figure] Figure 5 : Representative relative-price trajectories across consecutive, non-overlapping forecast blocks. The left and right columns show TAFAS and PETSA, respectively. The dark gray line denotes the ground truth, the red line denotes the frozen base model, the blue line denotes the baseline TTA method, and the green line denotes the corresponding TTA method with DPAL. Vertical dotted lines mark the forecast origins. Each price trajectory is initialized to $1$ only once at the beginning of the full sequence and is then accumulated across non-overlapping $H$ -day forecast blocks.

In the displayed cases, trajectories produced with DPAL often remain closer to the observed price paths over successive forecast blocks. Compared with the base model, TAFAS and PETSA, the corresponding DPAL variants often show less cumulative drift. These qualitative patterns are consistent with the reductions in reconstructed-price MSE reported in the preceding tables, although the magnitude of the improvement varies across experimental settings.

#### 4.3 Robustness to Loss Coefficients

To examine the robustness of DPAL to the choice of $\alpha$ and $\beta$ , we evaluate $\alpha\in\{10^{3},10^{4},10^{5},10^{6},10^{7}\}$ and $\beta\in\{10^{0},10^{1},10^{2},10^{3},10^{4}\}$ . Across all evaluated settings, we evaluate every $(\alpha,\beta)$ configuration while keeping all other training and evaluation settings unchanged.

Because reconstructed-price MSE can differ substantially in scale across datasets and forecasting settings, directly averaging raw MSE values would give disproportionate influence to settings with larger errors. We therefore compare each $(\alpha,\beta)$ configuration with the common configuration $(\alpha,\beta)=(10^{5},10^{2})$ used in the main experiments. For each setting $s$ , the relative change is defined as

$$\Delta_{s}(\alpha,\beta)=\frac{\mathrm{MSE}_{s}(\alpha,\beta)}{\mathrm{MSE}_{s}(10^{5},10^{2})}-1.$$ (11)

For each TTA framework, every table entry reports the median of $\Delta_{s}(\alpha,\beta)$ across all evaluated settings. Positive values indicate higher reconstructed-price MSE than the common configuration and therefore worse performance, whereas negative values indicate lower MSE and better performance. We use the median to reduce the influence of a small number of settings with unusually large relative changes.

As shown in Table 3 , small values of $\alpha$ , particularly $\alpha=10^{3}$ , result in substantially higher relative MSE under both TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) . In contrast, most configurations with $\alpha\geq 10^{5}$ perform similarly to the common configuration used in the main experiments. Although a few configurations yield slight improvements, the median improvements remain below $0.5\%$ . Overall, the results indicate a broad region of stable performance around the coefficient scales used in the main experiments.

Table: Table 3 : Robustness of DPAL to the loss coefficients $\alpha$ and $\beta$ under (a) TAFAS and (b) PETSA. Each cell reports the median relative change in reconstructed-price MSE, expressed as a percentage, from the common configuration $(\alpha,\beta)=(10^{5},10^{2})$ used in the main experiments across all evaluated settings within the corresponding TTA framework. Positive values indicate higher MSE and worse performance relative to the common configuration, whereas negative values indicate lower MSE and better performance. The outlined cells identify the common configuration, and the red and green shading denote performance degradation and improvement, respectively.

| (a) TAFAS |
| Median MSE change relative to the common configuration |
| $\bm{\alpha\backslash\beta}$ | $10^{0}$ | $10^{1}$ | $10^{2}$ | $10^{3}$ | $10^{4}$ |
| $10^{3}$ | 20.01% | 18.43% | 23.05% | 22.75% | 22.63% |
| $10^{4}$ | 4.58% | 3.90% | 7.46% | 13.25% | 14.73% |
| $10^{5}$ | 0.66% | 0.61% | 0.00% | 2.46% | 7.80% |
| $10^{6}$ | 0.27% | 0.22% | 0.12% | -0.43% | 2.07% |
| $10^{7}$ | 0.16% | 0.17% | 0.17% | 0.03% | -0.39% |
| (b) PETSA |
| Median MSE change relative to the common configuration |
| $\bm{\alpha\backslash\beta}$ | $10^{0}$ | $10^{1}$ | $10^{2}$ | $10^{3}$ | $10^{4}$ |
| $10^{3}$ | 13.06% | 11.93% | 13.51% | 15.83% | 15.50% |
| $10^{4}$ | 2.63% | 1.99% | 3.87% | 7.90% | 8.81% |
| $10^{5}$ | 0.62% | 0.50% | 0.00% | 1.90% | 5.79% |
| $10^{6}$ | 0.30% | 0.21% | 0.24% | -0.18% | 1.77% |
| $10^{7}$ | 0.19% | 0.07% | 0.09% | 0.23% | -0.17% |

To further assess the robustness of the common configuration, we compare it with the lowest MSE obtained for each evaluated setting across all coefficient configurations. Specifically, we define

$$G_{s}=\frac{\mathrm{MSE}_{s}(10^{5},10^{2})}{\min_{\alpha,\beta}\mathrm{MSE}_{s}(\alpha,\beta)}-1.$$ (12)

Table: Table 4 : Comparison between the common configuration $(\alpha,\beta)=(10^{5},10^{2})$ and the lowest-MSE configuration for each setting. The third column reports the number of settings with $G_{s}\leq 0.05$ , corresponding to a reconstructed-price MSE no more than 5% higher than the lowest MSE.

| Framework | Settings | $G_{s}\leq 0.05$ | Share | Median $G_{s}$ |
| TAFAS | 120 | 108 | 90.0% | 0.98% |
| PETSA | 120 | 94 | 78.3% | 1.54% |
| Combined | 240 | 202 | 84.2% | 1.30% |

We regard the common configuration as being close to the lowest-MSE configuration for setting $s$ when $G_{s}\leq 0.05$ . For each TTA framework, the evaluation covers eight datasets, five forecasting backbones, and three prediction horizons. Table 4 summarizes the number and share of settings satisfying this criterion, together with the median $G_{s}$ .

Across all evaluated settings, the common configuration is within 5% of the lowest MSE in 84.2% of cases, with a median $G_{s}$ of 1.30%. These results indicate that DPAL generally remains robust to the choice of loss coefficients.

### 5 Conclusions, Limitations, and Future Work

Conclusions. This work introduces Direction- and Price-Aware Adaptation Loss (DPAL), an adaptation objective for financial time series forecasting that combines pointwise return fitting with directional and price-calibration terms. DPAL is used as the adaptation objective in TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) without modifying the source forecaster or the adapter architecture. Across the evaluated settings, using DPAL reduces reconstructed-price MSE in most configurations for both TAFAS and PETSA. The rolling price trajectories further show that the corresponding DPAL variants often stay closer to the ground-truth price paths with less cumulative drift over successive forecast blocks. Overall, these results indicate that DPAL can improve test-time adaptation for financial time series forecasting under the evaluated settings.

Limitations. Although the evaluation covers multiple financial datasets, it remains limited to the markets and time periods represented in the collected data. In addition, the current formulation of DPAL uses relatively simple directional and price-calibration terms.

Future Work. Future work could extend the evaluation to broader market regimes and longer time periods, and explore more flexible directional and price-calibration terms. DPAL could also be studied in higher-frequency financial forecasting settings, where return dynamics and adaptation timing may differ from those considered here.

### Appendix A Additional Experimental Results

#### A.1 Ablation Study

To examine the individual contributions of the proposed direction-aware loss ( $\mathcal{L}_{\mathrm{dir}}$ ) and price calibration loss ( $\mathcal{L}_{\mathrm{price}}$ ), we conducted an ablation study. We evaluated all possible combinations of the three loss components. Specifically, we tested seven loss configurations: standard Mean Squared Error alone ( $\mathrm{MSE}$ ), direction-aware loss alone ( $\alpha\mathcal{L}_{\mathrm{dir}}$ ), price calibration loss alone ( $\beta\mathcal{L}_{\mathrm{price}}$ ), their pairwise combinations ( $\mathrm{MSE}+\alpha\mathcal{L}_{\mathrm{dir}}$ , $\mathrm{MSE}+\beta\mathcal{L}_{\mathrm{price}}$ , $\alpha\mathcal{L}_{\mathrm{dir}}+\beta\mathcal{L}_{\mathrm{price}}$ ), and the full joint formulation ( $\mathrm{MSE}+\alpha\mathcal{L}_{\mathrm{dir}}+\beta\mathcal{L}_{\mathrm{price}}$ ). These configurations were evaluated across different datasets, backbone models, and forecast horizons under both the TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) frameworks. Tables A.1.1 – A.1.1 report the results.

Table: Table A.1 : Ablation results for different combinations of the loss components under the TAFAS and PETSA frameworks. Base denotes the frozen backbone without test-time adaptation. The check-marked columns represent the seven loss configurations. In the PETSA tables, the additional column labeled original reports results using PETSA’s original adaptation objective. Each entry reports the reconstructed price MSE for a given backbone model and forecast horizon $H$ , with lower values indicating better performance. The best result (lowest MSE) in each row is highlighted in green, and the second-best result (second-lowest MSE) is marked in blue. All values are rounded to five decimal places, whereas the rankings are determined using the original unrounded results.

| Model | $H$ | Base | mse | $\checkmark$ |  |  | $\checkmark$ | $\checkmark$ |  | $\checkmark$ |
| dir |  | $\checkmark$ |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ |
| price |  |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| iTransformer | 60 | 1.61743 |  | 1.13657 | 0.69084 | 1.01203 | 0.54975 | 0.74609 | 0.54121 | 0.54736 |
| 90 | 6.16393 |  | 3.63401 | 1.86856 | 2.85430 | 1.51084 | 2.20571 | 1.47397 | 1.48232 |
| 120 | 17.71737 |  | 9.08681 | 4.47373 | 8.62511 | 2.80610 | 5.23148 | 2.77784 | 2.77978 |
| DLinear | 60 | 0.75416 |  | 0.74219 | 0.51022 | 0.70659 | 0.47830 | 0.69959 | 0.47640 | 0.47661 |
| 90 | 2.06209 |  | 1.99329 | 1.10354 | 1.54488 | 0.98637 | 1.52346 | 0.94325 | 0.94340 |
| 120 | 3.87860 |  | 3.68547 | 1.95164 | 3.06590 | 1.52380 | 2.93949 | 1.49955 | 1.50322 |
| FreTS | 60 | 0.52950 |  | 0.52818 | 0.48641 | 0.50994 | 0.47605 | 0.50921 | 0.47509 | 0.47602 |
| 90 | 1.15248 |  | 1.15159 | 1.01343 | 1.09132 | 0.94249 | 1.08163 | 0.93252 | 0.93348 |
| 120 | 2.03039 |  | 2.03178 | 1.82891 | 1.98825 | 1.57614 | 1.95812 | 1.60539 | 1.61589 |
| PatchTST | 60 | 1.50215 |  | 0.76392 | 0.60709 | 0.93288 | 0.49720 | 0.54765 | 0.49001 | 0.48863 |
| 90 | 5.26547 |  | 2.29294 | 1.61467 | 2.07171 | 1.19600 | 1.25270 | 1.14013 | 1.13853 |
| 120 | 13.14714 |  | 5.50601 | 3.52361 | 6.08743 | 2.15644 | 3.06051 | 2.07796 | 2.07467 |
| OLS | 60 | 1.78386 |  | 1.11830 | 0.61121 | 0.89851 | 0.52656 | 0.69139 | 0.51812 | 0.51547 |
| 90 | 5.76070 |  | 2.86407 | 1.59095 | 1.97215 | 1.38104 | 1.53145 | 1.38284 | 1.37908 |
| 120 | 15.58452 |  | 6.79152 | 3.84079 | 6.31496 | 2.92593 | 4.29467 | 3.01510 | 3.00353 |

Table: A.1.1 TAFAS S&P 500 Excerpt

| Model | $H$ | Base | mse | $\checkmark$ |  |  | $\checkmark$ | $\checkmark$ |  | $\checkmark$ |
| dir |  | $\checkmark$ |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ |
| price |  |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| iTransformer | 60 | 0.31049 |  | 0.23600 | 0.13022 | 0.18126 | 0.10350 | 0.13573 | 0.10346 | 0.10370 |
| 90 | 1.08383 |  | 0.75797 | 0.32101 | 0.57364 | 0.28542 | 0.45288 | 0.28627 | 0.28569 |
| 120 | 3.66154 |  | 2.50056 | 0.92215 | 2.32753 | 0.88228 | 1.72442 | 0.89078 | 0.88937 |
| DLinear | 60 | 0.12715 |  | 0.12572 | 0.09147 | 0.11144 | 0.08590 | 0.10870 | 0.08572 | 0.08573 |
| 90 | 0.30746 |  | 0.29315 | 0.17772 | 0.22061 | 0.16674 | 0.21476 | 0.16449 | 0.16436 |
| 120 | 0.61329 |  | 0.59441 | 0.31469 | 0.45443 | 0.29504 | 0.43984 | 0.28896 | 0.28857 |
| FreTS | 60 | 0.07956 |  | 0.07926 | 0.07572 | 0.07905 | 0.07489 | 0.07914 | 0.07518 | 0.07525 |
| 90 | 0.15343 |  | 0.15248 | 0.14383 | 0.15360 | 0.13732 | 0.15337 | 0.13696 | 0.13695 |
| 120 | 0.27454 |  | 0.27336 | 0.25104 | 0.27378 | 0.23120 | 0.27318 | 0.23000 | 0.23053 |
| PatchTST | 60 | 0.28742 |  | 0.13806 | 0.09950 | 0.16196 | 0.09453 | 0.09919 | 0.09454 | 0.09428 |
| 90 | 0.79353 |  | 0.31934 | 0.19844 | 0.48195 | 0.19618 | 0.27224 | 0.19784 | 0.19696 |
| 120 | 2.41472 |  | 0.73501 | 0.44984 | 1.54950 | 0.48298 | 0.70732 | 0.50250 | 0.49946 |
| OLS | 60 | 0.36039 |  | 0.23918 | 0.13081 | 0.20259 | 0.11541 | 0.15304 | 0.11533 | 0.11527 |
| 90 | 1.18749 |  | 0.71727 | 0.34031 | 0.69211 | 0.31792 | 0.52828 | 0.31934 | 0.31922 |
| 120 | 4.10791 |  | 2.48495 | 0.98300 | 3.34456 | 0.96449 | 2.53727 | 0.97768 | 0.97911 |
| Model | $H$ | Base | mse | $\checkmark$ |  |  | $\checkmark$ | $\checkmark$ |  | $\checkmark$ |
| dir |  | $\checkmark$ |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ |
| price |  |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| iTransformer | 60 | 0.30491 |  | 0.22266 | 0.10367 | 0.22091 | 0.09676 | 0.14446 | 0.09657 | 0.10505 |
| 90 | 0.69923 |  | 0.40823 | 0.21734 | 0.40426 | 0.16352 | 0.27378 | 0.15242 | 0.15310 |
| 120 | 1.45384 |  | 0.91903 | 0.50233 | 1.02163 | 0.35110 | 0.73352 | 0.33515 | 0.33906 |
| DLinear | 60 | 0.12283 |  | 0.12568 | 0.08020 | 0.11390 | 0.07334 | 0.11505 | 0.07317 | 0.07326 |
| 90 | 0.29572 |  | 0.28654 | 0.17123 | 0.23883 | 0.14525 | 0.23303 | 0.14292 | 0.14326 |
| 120 | 0.50299 |  | 0.49319 | 0.30670 | 0.42453 | 0.26384 | 0.41608 | 0.26156 | 0.26247 |
| FreTS | 60 | 0.06752 |  | 0.06948 | 0.06519 | 0.06822 | 0.06544 | 0.06997 | 0.06581 | 0.06588 |
| 90 | 0.14269 |  | 0.14350 | 0.13450 | 0.14204 | 0.12784 | 0.14292 | 0.12694 | 0.12726 |
| 120 | 0.24604 |  | 0.24637 | 0.23695 | 0.24966 | 0.22598 | 0.25160 | 0.22461 | 0.22559 |
| PatchTST | 60 | 0.23714 |  | 0.10034 | 0.08920 | 0.16727 | 0.07233 | 0.08412 | 0.07150 | 0.07135 |
| 90 | 0.62163 |  | 0.23122 | 0.20489 | 0.36289 | 0.14232 | 0.17970 | 0.13505 | 0.13505 |
| 120 | 1.19838 |  | 0.53473 | 0.45986 | 0.76042 | 0.30306 | 0.37024 | 0.27931 | 0.27867 |
| OLS | 60 | 0.30977 |  | 0.14949 | 0.09819 | 0.14626 | 0.07953 | 0.11407 | 0.08022 | 0.08062 |
| 90 | 0.79484 |  | 0.35215 | 0.22567 | 0.21361 | 0.15854 | 0.19225 | 0.14915 | 0.14956 |
| 120 | 1.58614 |  | 0.83380 | 0.50402 | 0.50750 | 0.34201 | 0.44580 | 0.33023 | 0.33076 |

Table: A.1.1 TAFAS Technology Sector

| Model | $H$ | Base | mse | $\checkmark$ |  |  | $\checkmark$ | $\checkmark$ |  | $\checkmark$ |
| dir |  | $\checkmark$ |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ |
| price |  |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| iTransformer | 60 | 0.42161 |  | 0.21446 | 0.10671 | 0.19210 | 0.10269 | 0.15871 | 0.10198 | 0.10243 |
| 90 | 1.26285 |  | 0.51497 | 0.22179 | 0.37666 | 0.19041 | 0.29334 | 0.18713 | 0.18650 |
| 120 | 2.85160 |  | 1.31550 | 0.46201 | 0.94775 | 0.32796 | 0.67012 | 0.31184 | 0.31325 |
| DLinear | 60 | 0.16194 |  | 0.15340 | 0.10161 | 0.14902 | 0.09750 | 0.14047 | 0.09675 | 0.09652 |
| 90 | 0.35196 |  | 0.33752 | 0.19132 | 0.24699 | 0.17665 | 0.24529 | 0.17148 | 0.17167 |
| 120 | 0.59787 |  | 0.58418 | 0.32343 | 0.43689 | 0.27900 | 0.44063 | 0.27600 | 0.27659 |
| FreTS | 60 | 0.09694 |  | 0.09827 | 0.08959 | 0.09614 | 0.08929 | 0.09680 | 0.08925 | 0.08936 |
| 90 | 0.18422 |  | 0.18591 | 0.16319 | 0.18401 | 0.15936 | 0.18556 | 0.15860 | 0.15911 |
| 120 | 0.30389 |  | 0.30402 | 0.26617 | 0.30341 | 0.25491 | 0.30671 | 0.25461 | 0.25510 |
| PatchTST | 60 | 0.30671 |  | 0.11481 | 0.10238 | 0.15836 | 0.09522 | 0.10280 | 0.09508 | 0.09506 |
| 90 | 0.83752 |  | 0.25486 | 0.19470 | 0.41181 | 0.17171 | 0.21413 | 0.17211 | 0.17237 |
| 120 | 1.77862 |  | 0.54692 | 0.37739 | 0.72563 | 0.29102 | 0.31431 | 0.27723 | 0.27726 |
| OLS | 60 | 0.46581 |  | 0.19646 | 0.10178 | 0.19697 | 0.10206 | 0.15371 | 0.10080 | 0.10170 |
| 90 | 1.26528 |  | 0.39689 | 0.21062 | 0.33343 | 0.18941 | 0.29275 | 0.19556 | 0.19723 |
| 120 | 2.85730 |  | 1.02155 | 0.43815 | 0.77485 | 0.33635 | 0.64332 | 0.32268 | 0.32373 |
| Model | $H$ | Base | mse | $\checkmark$ |  |  | $\checkmark$ | $\checkmark$ |  | $\checkmark$ |
| dir |  | $\checkmark$ |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ |
| price |  |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| iTransformer | 60 | 0.02593 |  | 0.02321 | 0.01470 | 0.02279 | 0.01293 | 0.01951 | 0.01285 | 0.01287 |
| 90 | 0.05730 |  | 0.05107 | 0.03065 | 0.04711 | 0.02576 | 0.04303 | 0.02558 | 0.02580 |
| 120 | 0.11261 |  | 0.10216 | 0.05875 | 0.10520 | 0.04693 | 0.09472 | 0.04752 | 0.04851 |
| DLinear | 60 | 0.01388 |  | 0.01410 | 0.01197 | 0.01372 | 0.01145 | 0.01391 | 0.01142 | 0.01139 |
| 90 | 0.02799 |  | 0.02798 | 0.02277 | 0.02606 | 0.02125 | 0.02608 | 0.02086 | 0.02086 |
| 120 | 0.04618 |  | 0.04645 | 0.03707 | 0.04391 | 0.03451 | 0.04336 | 0.03395 | 0.03391 |
| FreTS | 60 | 0.00996 |  | 0.01002 | 0.00966 | 0.00991 | 0.00982 | 0.01009 | 0.00983 | 0.00982 |
| 90 | 0.01801 |  | 0.01819 | 0.01721 | 0.01820 | 0.01735 | 0.01853 | 0.01727 | 0.01727 |
| 120 | 0.02909 |  | 0.02958 | 0.02760 | 0.02915 | 0.02756 | 0.02936 | 0.02745 | 0.02751 |
| PatchTST | 60 | 0.02102 |  | 0.01554 | 0.01329 | 0.01831 | 0.01176 | 0.01409 | 0.01171 | 0.01175 |
| 90 | 0.04354 |  | 0.03153 | 0.02697 | 0.03689 | 0.02340 | 0.03051 | 0.02380 | 0.02394 |
| 120 | 0.08100 |  | 0.05787 | 0.04912 | 0.07020 | 0.04199 | 0.05679 | 0.04261 | 0.04290 |
| OLS | 60 | 0.02602 |  | 0.02171 | 0.01512 | 0.02208 | 0.01325 | 0.01864 | 0.01322 | 0.01325 |
| 90 | 0.05850 |  | 0.04898 | 0.03197 | 0.04941 | 0.02890 | 0.04329 | 0.02733 | 0.02732 |
| 120 | 0.11241 |  | 0.09377 | 0.06099 | 0.14025 | 0.05283 | 0.10597 | 0.05183 | 0.05142 |

Table: A.1.1 TAFAS Energy Sector

| Model | $H$ | Base | mse | $\checkmark$ |  |  | $\checkmark$ | $\checkmark$ |  | $\checkmark$ |
| dir |  | $\checkmark$ |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ |
| price |  |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| iTransformer | 60 | 0.03685 |  | 0.03582 | 0.02889 | 0.03232 | 0.02420 | 0.03052 | 0.02388 | 0.02394 |
| 90 | 0.08183 |  | 0.07929 | 0.05959 | 0.06565 | 0.04919 | 0.06332 | 0.04783 | 0.04792 |
| 120 | 0.14979 |  | 0.14584 | 0.10643 | 0.12253 | 0.08851 | 0.11775 | 0.08626 | 0.08630 |
| DLinear | 60 | 0.02703 |  | 0.02706 | 0.02389 | 0.02606 | 0.02270 | 0.02614 | 0.02259 | 0.02259 |
| 90 | 0.05452 |  | 0.05439 | 0.04653 | 0.05057 | 0.04460 | 0.05084 | 0.04376 | 0.04377 |
| 120 | 0.09314 |  | 0.09334 | 0.07890 | 0.08959 | 0.07359 | 0.08954 | 0.07240 | 0.07256 |
| FreTS | 60 | 0.01859 |  | 0.01862 | 0.01876 | 0.01842 | 0.01905 | 0.01855 | 0.01907 | 0.01907 |
| 90 | 0.03513 |  | 0.03515 | 0.03553 | 0.03509 | 0.03608 | 0.03520 | 0.03617 | 0.03619 |
| 120 | 0.05924 |  | 0.05927 | 0.05937 | 0.05910 | 0.05963 | 0.05928 | 0.05958 | 0.05960 |
| PatchTST | 60 | 0.03588 |  | 0.02961 | 0.02611 | 0.03071 | 0.02307 | 0.02603 | 0.02298 | 0.02299 |
| 90 | 0.07545 |  | 0.05974 | 0.05193 | 0.06025 | 0.04637 | 0.05237 | 0.04585 | 0.04581 |
| 120 | 0.14276 |  | 0.10966 | 0.08976 | 0.11428 | 0.08681 | 0.09559 | 0.08566 | 0.08548 |
| OLS | 60 | 0.03871 |  | 0.03468 | 0.02814 | 0.03288 | 0.02420 | 0.03084 | 0.02402 | 0.02410 |
| 90 | 0.08505 |  | 0.07633 | 0.05805 | 0.06563 | 0.04952 | 0.06329 | 0.04897 | 0.04900 |
| 120 | 0.15842 |  | 0.14309 | 0.10604 | 0.12391 | 0.09186 | 0.12003 | 0.09043 | 0.09056 |
| Model | $H$ | Base | mse | $\checkmark$ |  |  | $\checkmark$ | $\checkmark$ |  | $\checkmark$ |
| dir |  | $\checkmark$ |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ |
| price |  |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| iTransformer | 60 | 0.05752 |  | 0.05120 | 0.03579 | 0.04748 | 0.03150 | 0.04126 | 0.03117 | 0.03120 |
| 90 | 0.12469 |  | 0.11041 | 0.07129 | 0.09201 | 0.06084 | 0.08366 | 0.05923 | 0.05948 |
| 120 | 0.22275 |  | 0.19713 | 0.12478 | 0.17644 | 0.10536 | 0.15893 | 0.10383 | 0.10428 |
| DLinear | 60 | 0.03478 |  | 0.03487 | 0.03001 | 0.03394 | 0.02854 | 0.03369 | 0.02839 | 0.02835 |
| 90 | 0.07097 |  | 0.06953 | 0.05640 | 0.06453 | 0.05199 | 0.06423 | 0.05073 | 0.05062 |
| 120 | 0.11325 |  | 0.11182 | 0.08922 | 0.10578 | 0.08064 | 0.10431 | 0.07868 | 0.07857 |
| FreTS | 60 | 0.02635 |  | 0.02644 | 0.02542 | 0.02614 | 0.02572 | 0.02623 | 0.02572 | 0.02569 |
| 90 | 0.04852 |  | 0.04852 | 0.04512 | 0.04807 | 0.04473 | 0.04835 | 0.04445 | 0.04447 |
| 120 | 0.07669 |  | 0.07689 | 0.07040 | 0.07618 | 0.06878 | 0.07634 | 0.06835 | 0.06843 |
| PatchTST | 60 | 0.05240 |  | 0.03595 | 0.03423 | 0.04295 | 0.02906 | 0.03036 | 0.02885 | 0.02879 |
| 90 | 0.10923 |  | 0.07137 | 0.06663 | 0.07794 | 0.05439 | 0.05619 | 0.05309 | 0.05309 |
| 120 | 0.19164 |  | 0.12389 | 0.11390 | 0.14208 | 0.09288 | 0.09489 | 0.08997 | 0.08985 |
| OLS | 60 | 0.05930 |  | 0.04541 | 0.03629 | 0.04705 | 0.03213 | 0.03849 | 0.03196 | 0.03198 |
| 90 | 0.12777 |  | 0.09432 | 0.07364 | 0.08609 | 0.06378 | 0.07969 | 0.06323 | 0.06332 |
| 120 | 0.22788 |  | 0.16949 | 0.12910 | 0.15975 | 0.10747 | 0.14324 | 0.10574 | 0.10652 |

Table: A.1.1 TAFAS Macro and Market Series

| Model | $H$ | Base | mse | $\checkmark$ |  |  | $\checkmark$ | $\checkmark$ |  | $\checkmark$ |
| dir |  | $\checkmark$ |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ |
| price |  |  | $\checkmark$ |  | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| iTransformer | 60 | 0.01886 |  | 0.01606 | 0.01207 | 0.01567 | 0.01099 | 0.01315 | 0.01097 | 0.01092 |
| 90 | 0.04027 |  | 0.03476 | 0.02344 | 0.02956 | 0.02199 | 0.02661 | 0.02124 | 0.02116 |
| 120 | 0.07258 |  | 0.06056 | 0.03946 | 0.05605 | 0.03702 | 0.05049 | 0.03600 | 0.03579 |
| DLinear | 60 | 0.01248 |  | 0.01228 | 0.01101 | 0.01226 | 0.01062 | 0.01190 | 0.01059 | 0.01057 |
| 90 | 0.02567 |  | 0.02484 | 0.02159 | 0.02317 | 0.02052 | 0.02230 | 0.02014 | 0.02012 |
| 120 | 0.04046 |  | 0.03971 | 0.03356 | 0.03792 | 0.03147 | 0.03694 | 0.03116 | 0.03101 |
| FreTS | 60 | 0.00882 |  | 0.00883 | 0.00910 | 0.00880 | 0.00938 | 0.00897 | 0.00940 | 0.00936 |
| 90 | 0.01670 |  | 0.01673 | 0.01704 | 0.01665 | 0.01743 | 0.01667 | 0.01726 | 0.01722 |
| 120 | 0.02586 |  | 0.02594 | 0.02614 | 0.02593 | 0.02659 | 0.02589 | 0.02645 | 0.02642 |
| PatchTST | 60 | 0.01793 |  | 0.01208 | 0.01152 | 0.01388 | 0.01052 | 0.01029 | 0.01045 | 0.01038 |
| 90 | 0.03668 |  | 0.02444 | 0.02201 | 0.02495 | 0.01957 | 0.02000 | 0.01879 | 0.01869 |
| 120 | 0.06088 |  | 0.04017 | 0.03578 | 0.04431 | 0.03253 | 0.03414 | 0.03106 | 0.03086 |
| OLS | 60 | 0.02018 |  | 0.01473 | 0.01207 | 0.01430 | 0.01097 | 0.01205 | 0.01097 | 0.01087 |
| 90 | 0.04483 |  | 0.03259 | 0.02399 | 0.02716 | 0.02192 | 0.02361 | 0.02155 | 0.02150 |
| 120 | 0.07908 |  | 0.05686 | 0.03924 | 0.04825 | 0.03570 | 0.04205 | 0.03484 | 0.03522 |
| Model | $H$ | Base | original | mse | ✓ |  |  | ✓ | ✓ |  | ✓ |
| dir |  | ✓ |  | ✓ |  | ✓ | ✓ |
| price |  |  | ✓ |  | ✓ | ✓ | ✓ |
| iTransformer | 60 | 1.61743 | 1.07880 |  | 1.00403 | 0.68592 | 1.06174 | 0.68473 | 0.81383 | 0.66979 | 0.78249 |
| 90 | 6.16393 | 3.45883 |  | 2.94524 | 1.66862 | 3.15973 | 1.66476 | 2.28144 | 1.56710 | 1.84711 |
| 120 | 17.71737 | 7.58773 |  | 7.77118 | 3.60424 | 10.26046 | 3.63848 | 6.76601 | 3.42331 | 3.89974 |
| DLinear | 60 | 0.75416 | 0.76475 |  | 0.76093 | 0.49869 | 0.66956 | 0.49986 | 0.68791 | 0.49920 | 0.50238 |
| 90 | 2.06209 | 1.82557 |  | 1.71048 | 1.03138 | 1.50014 | 1.03347 | 1.46148 | 0.98851 | 1.01324 |
| 120 | 3.87860 | 3.60260 |  | 3.49313 | 1.70534 | 3.25740 | 1.70859 | 3.06473 | 1.69181 | 1.55223 |
| FreTS | 60 | 0.52950 | 0.57772 |  | 0.55204 | 0.46512 | 0.50428 | 0.46569 | 0.52456 | 0.46467 | 0.46929 |
| 90 | 1.15247 | 1.19871 |  | 1.18350 | 0.92256 | 1.06746 | 0.92406 | 1.06852 | 0.92024 | 0.88769 |
| 120 | 2.03039 | 2.14219 |  | 2.01618 | 1.53752 | 1.96713 | 1.54297 | 1.91395 | 1.56167 | 1.46289 |
| PatchTST | 60 | 1.50215 | 0.70887 |  | 0.65572 | 0.51385 | 0.98248 | 0.51307 | 0.62744 | 0.50344 | 0.48915 |
| 90 | 5.26546 | 1.96757 |  | 2.02453 | 1.37594 | 2.43478 | 1.37078 | 1.65070 | 1.27632 | 1.10568 |
| 120 | 13.14714 | 5.28275 |  | 5.64375 | 2.77521 | 7.35952 | 2.79816 | 4.40622 | 2.71818 | 2.65405 |
| OLS | 60 | 1.78386 | 1.10668 |  | 0.82033 | 0.59045 | 1.09650 | 0.58908 | 0.81343 | 0.57098 | 0.64600 |
| 90 | 5.76070 | 2.45446 |  | 1.99504 | 1.45515 | 2.61408 | 1.45487 | 2.14437 | 1.33146 | 1.29933 |
| 120 | 15.58452 | 5.35221 |  | 5.18654 | 3.06482 | 7.08726 | 3.07645 | 5.58615 | 3.08803 | 2.78337 |

Table: A.1.1 PETSA S&P 500 Excerpt

| Model | $H$ | Base | original | mse | ✓ |  |  | ✓ | ✓ |  | ✓ |
| dir |  | ✓ |  | ✓ |  | ✓ | ✓ |
| price |  |  | ✓ |  | ✓ | ✓ | ✓ |
| iTransformer | 60 | 0.31049 | 0.17225 |  | 0.16791 | 0.11173 | 0.19314 | 0.11210 | 0.15084 | 0.11315 | 0.12200 |
| 90 | 1.08383 | 0.43358 |  | 0.50458 | 0.29117 | 0.47271 | 0.29077 | 0.37550 | 0.28899 | 0.31976 |
| 120 | 3.66154 | 1.32155 |  | 1.56566 | 0.89284 | 1.74055 | 0.89279 | 1.24945 | 0.88830 | 0.96333 |
| DLinear | 60 | 0.12715 | 0.14383 |  | 0.13382 | 0.09187 | 0.12159 | 0.09231 | 0.12473 | 0.09114 | 0.08410 |
| 90 | 0.30746 | 0.32153 |  | 0.29276 | 0.19141 | 0.24487 | 0.19166 | 0.23608 | 0.18748 | 0.16397 |
| 120 | 0.61329 | 0.68129 |  | 0.60700 | 0.32525 | 0.47748 | 0.32623 | 0.44502 | 0.30402 | 0.28872 |
| FreTS | 60 | 0.07956 | 0.08676 |  | 0.08135 | 0.07416 | 0.07883 | 0.07428 | 0.08018 | 0.07434 | 0.07487 |
| 90 | 0.15343 | 0.16424 |  | 0.15190 | 0.13559 | 0.15249 | 0.13590 | 0.15004 | 0.13669 | 0.13526 |
| 120 | 0.27454 | 0.30168 |  | 0.27128 | 0.23052 | 0.27221 | 0.23106 | 0.26629 | 0.23154 | 0.22538 |
| PatchTST | 60 | 0.28742 | 0.10828 |  | 0.10547 | 0.09665 | 0.17528 | 0.09671 | 0.10784 | 0.09616 | 0.10309 |
| 90 | 0.79353 | 0.21113 |  | 0.21860 | 0.20131 | 0.40784 | 0.20229 | 0.24826 | 0.20430 | 0.19987 |
| 120 | 2.41471 | 0.55255 |  | 0.56534 | 0.45407 | 1.23980 | 0.45376 | 0.61070 | 0.45336 | 0.46982 |
| OLS | 60 | 0.36039 | 0.17976 |  | 0.16889 | 0.11857 | 0.20191 | 0.11861 | 0.15114 | 0.11886 | 0.12514 |
| 90 | 1.18749 | 0.42979 |  | 0.42494 | 0.31405 | 0.43555 | 0.31237 | 0.37003 | 0.30813 | 0.33303 |
| 120 | 4.10791 | 1.26770 |  | 1.22648 | 0.95783 | 1.46176 | 0.95624 | 1.18374 | 0.93823 | 0.94652 |
| Model | $H$ | Base | original | mse | ✓ |  |  | ✓ | ✓ |  | ✓ |
| dir |  | ✓ |  | ✓ |  | ✓ | ✓ |
| price |  |  | ✓ |  | ✓ | ✓ | ✓ |
| iTransformer | 60 | 0.30491 | 0.19194 |  | 0.22250 | 0.11596 | 0.26009 | 0.11825 | 0.17019 | 0.11497 | 0.11831 |
| 90 | 0.69923 | 0.61013 |  | 0.25800 | 0.21674 | 0.45696 | 0.21689 | 0.29670 | 0.20746 | 0.15115 |
| 120 | 1.45384 | 0.76688 |  | 0.62010 | 0.49672 | 1.04817 | 0.49697 | 0.67196 | 0.47574 | 0.36467 |
| DLinear | 60 | 0.12283 | 0.13785 |  | 0.15255 | 0.07802 | 0.11757 | 0.07831 | 0.11417 | 0.07846 | 0.07524 |
| 90 | 0.29572 | 0.24089 |  | 0.30426 | 0.16120 | 0.22626 | 0.16167 | 0.22808 | 0.16031 | 0.13707 |
| 120 | 0.50299 | 0.42464 |  | 0.52829 | 0.28865 | 0.44771 | 0.28960 | 0.44231 | 0.28291 | 0.25080 |
| FreTS | 60 | 0.06752 | 0.08139 |  | 0.07521 | 0.06437 | 0.06948 | 0.06453 | 0.07434 | 0.06459 | 0.06536 |
| 90 | 0.14269 | 0.16834 |  | 0.15153 | 0.13127 | 0.14327 | 0.13163 | 0.14855 | 0.13152 | 0.12702 |
| 120 | 0.24603 | 0.28902 |  | 0.25990 | 0.22726 | 0.25255 | 0.22789 | 0.26295 | 0.22891 | 0.22538 |
| PatchTST | 60 | 0.23714 | 0.10576 |  | 0.10338 | 0.08588 | 0.16036 | 0.08608 | 0.09266 | 0.08544 | 0.08393 |
| 90 | 0.62163 | 0.21849 |  | 0.22426 | 0.19151 | 0.36987 | 0.19194 | 0.22143 | 0.18946 | 0.14741 |
| 120 | 1.19838 | 0.44963 |  | 0.48724 | 0.41816 | 0.79983 | 0.41925 | 0.46406 | 0.41218 | 0.29811 |
| OLS | 60 | 0.30977 | 0.14224 |  | 0.13248 | 0.08888 | 0.13314 | 0.08926 | 0.11117 | 0.08863 | 0.08327 |
| 90 | 0.79484 | 0.28714 |  | 0.23213 | 0.19452 | 0.28532 | 0.19585 | 0.23270 | 0.19405 | 0.16022 |
| 120 | 1.58614 | 0.63182 |  | 0.59956 | 0.44173 | 0.73436 | 0.44528 | 0.56560 | 0.43284 | 0.33070 |

Table: A.1.1 PETSA Technology Sector

| Model | $H$ | Base | original | mse | ✓ |  |  | ✓ | ✓ |  | ✓ |
| dir |  | ✓ |  | ✓ |  | ✓ | ✓ |
| price |  |  | ✓ |  | ✓ | ✓ | ✓ |
| iTransformer | 60 | 0.42161 | 0.19636 |  | 0.15864 | 0.09849 | 0.23867 | 0.09885 | 0.17437 | 0.09787 | 0.09929 |
| 90 | 1.26285 | 0.45928 |  | 0.37049 | 0.19947 | 0.49278 | 0.19969 | 0.32317 | 0.19629 | 0.19278 |
| 120 | 2.85160 | 0.93674 |  | 0.85556 | 0.40467 | 1.17880 | 0.40739 | 0.70321 | 0.39307 | 0.34146 |
| DLinear | 60 | 0.16194 | 0.21405 |  | 0.18530 | 0.09944 | 0.14540 | 0.10025 | 0.14566 | 0.09986 | 0.10094 |
| 90 | 0.35196 | 0.28986 |  | 0.36762 | 0.18323 | 0.23670 | 0.18316 | 0.23959 | 0.17729 | 0.17750 |
| 120 | 0.59787 | 0.65952 |  | 0.68040 | 0.30069 | 0.42510 | 0.30095 | 0.42508 | 0.30098 | 0.29276 |
| FreTS | 60 | 0.09694 | 0.11603 |  | 0.10637 | 0.08950 | 0.09647 | 0.08978 | 0.10346 | 0.08984 | 0.09096 |
| 90 | 0.18422 | 0.22629 |  | 0.19976 | 0.15830 | 0.18312 | 0.15869 | 0.19157 | 0.15958 | 0.15964 |
| 120 | 0.30389 | 0.39962 |  | 0.33023 | 0.25075 | 0.30568 | 0.25152 | 0.32093 | 0.25322 | 0.24903 |
| PatchTST | 60 | 0.30671 | 0.14063 |  | 0.12215 | 0.09916 | 0.17770 | 0.09908 | 0.11185 | 0.09848 | 0.10292 |
| 90 | 0.83752 | 0.28471 |  | 0.26085 | 0.18215 | 0.39180 | 0.18219 | 0.23400 | 0.18635 | 0.18498 |
| 120 | 1.77862 | 0.43336 |  | 0.48765 | 0.35324 | 0.82280 | 0.35699 | 0.38591 | 0.36388 | 0.32748 |
| OLS | 60 | 0.46581 | 0.20059 |  | 0.15660 | 0.10417 | 0.19130 | 0.10411 | 0.14755 | 0.10176 | 0.09960 |
| 90 | 1.26527 | 0.45351 |  | 0.31629 | 0.20398 | 0.30476 | 0.20029 | 0.26687 | 0.18750 | 0.17870 |
| 120 | 2.85730 | 0.94798 |  | 0.84752 | 0.38710 | 0.65648 | 0.38714 | 0.63713 | 0.36614 | 0.31308 |
| Model | $H$ | Base | original | mse | ✓ |  |  | ✓ | ✓ |  | ✓ |
| dir |  | ✓ |  | ✓ |  | ✓ | ✓ |
| price |  |  | ✓ |  | ✓ | ✓ | ✓ |
| iTransformer | 60 | 0.02593 | 0.02026 |  | 0.02009 | 0.01464 | 0.02327 | 0.01471 | 0.01956 | 0.01459 | 0.01308 |
| 90 | 0.05730 | 0.04530 |  | 0.04407 | 0.03117 | 0.04808 | 0.03144 | 0.04305 | 0.03163 | 0.02636 |
| 120 | 0.11261 | 0.08673 |  | 0.08694 | 0.05895 | 0.09730 | 0.05914 | 0.08430 | 0.05920 | 0.04684 |
| DLinear | 60 | 0.01388 | 0.01573 |  | 0.01441 | 0.01182 | 0.01373 | 0.01179 | 0.01422 | 0.01175 | 0.01121 |
| 90 | 0.02799 | 0.03065 |  | 0.02780 | 0.02172 | 0.02609 | 0.02167 | 0.02662 | 0.02132 | 0.02060 |
| 120 | 0.04618 | 0.04893 |  | 0.04699 | 0.03510 | 0.04453 | 0.03506 | 0.04531 | 0.03463 | 0.03308 |
| FreTS | 60 | 0.00996 | 0.01165 |  | 0.01022 | 0.00957 | 0.00985 | 0.00958 | 0.01025 | 0.00958 | 0.00970 |
| 90 | 0.01801 | 0.02189 |  | 0.01842 | 0.01675 | 0.01795 | 0.01676 | 0.01876 | 0.01675 | 0.01684 |
| 120 | 0.02909 | 0.03534 |  | 0.03009 | 0.02645 | 0.02917 | 0.02652 | 0.03063 | 0.02659 | 0.02641 |
| PatchTST | 60 | 0.02102 | 0.01636 |  | 0.01402 | 0.01274 | 0.01792 | 0.01272 | 0.01395 | 0.01272 | 0.01158 |
| 90 | 0.04354 | 0.03144 |  | 0.02790 | 0.02518 | 0.03430 | 0.02519 | 0.02846 | 0.02508 | 0.02236 |
| 120 | 0.08100 | 0.05415 |  | 0.05189 | 0.04500 | 0.06623 | 0.04500 | 0.05220 | 0.04456 | 0.03797 |
| OLS | 60 | 0.02602 | 0.02091 |  | 0.01894 | 0.01392 | 0.02072 | 0.01388 | 0.01908 | 0.01378 | 0.01276 |
| 90 | 0.05850 | 0.04311 |  | 0.03763 | 0.02990 | 0.03942 | 0.03002 | 0.03683 | 0.02971 | 0.02599 |
| 120 | 0.11241 | 0.07839 |  | 0.07447 | 0.05680 | 0.08246 | 0.05731 | 0.07439 | 0.05656 | 0.04572 |

Table: A.1.1 PETSA Energy Sector

| Model | $H$ | Base | original | mse | ✓ |  |  | ✓ | ✓ |  | ✓ |
| dir |  | ✓ |  | ✓ |  | ✓ | ✓ |
| price |  |  | ✓ |  | ✓ | ✓ | ✓ |
| iTransformer | 60 | 0.03685 | 0.03276 |  | 0.03380 | 0.02552 | 0.03316 | 0.02573 | 0.03207 | 0.02506 | 0.02497 |
| 90 | 0.08183 | 0.06904 |  | 0.07473 | 0.05257 | 0.06839 | 0.05297 | 0.06621 | 0.05097 | 0.05294 |
| 120 | 0.14979 | 0.12655 |  | 0.14045 | 0.09470 | 0.13382 | 0.09494 | 0.13085 | 0.09220 | 0.09400 |
| DLinear | 60 | 0.02703 | 0.02834 |  | 0.02827 | 0.02421 | 0.02626 | 0.02419 | 0.02687 | 0.02403 | 0.02324 |
| 90 | 0.05452 | 0.05819 |  | 0.05602 | 0.04829 | 0.05110 | 0.04821 | 0.05202 | 0.04665 | 0.04255 |
| 120 | 0.09314 | 0.10043 |  | 0.09706 | 0.08275 | 0.08971 | 0.08259 | 0.09164 | 0.08163 | 0.07527 |
| FreTS | 60 | 0.01859 | 0.01973 |  | 0.01900 | 0.01865 | 0.01823 | 0.01866 | 0.01878 | 0.01860 | 0.01886 |
| 90 | 0.03513 | 0.03833 |  | 0.03655 | 0.03552 | 0.03482 | 0.03556 | 0.03605 | 0.03545 | 0.03595 |
| 120 | 0.05924 | 0.06452 |  | 0.06152 | 0.05936 | 0.05863 | 0.05939 | 0.06076 | 0.05919 | 0.05933 |
| PatchTST | 60 | 0.03588 | 0.02903 |  | 0.02672 | 0.02491 | 0.03211 | 0.02492 | 0.03132 | 0.02461 | 0.02684 |
| 90 | 0.07545 | 0.05649 |  | 0.05616 | 0.04993 | 0.06565 | 0.04995 | 0.06229 | 0.04937 | 0.05191 |
| 120 | 0.14276 | 0.09991 |  | 0.10447 | 0.08882 | 0.12496 | 0.08890 | 0.11815 | 0.08832 | 0.08923 |
| OLS | 60 | 0.03871 | 0.03295 |  | 0.03161 | 0.02646 | 0.03129 | 0.02658 | 0.03033 | 0.02612 | 0.02704 |
| 90 | 0.08505 | 0.07124 |  | 0.06834 | 0.05419 | 0.06361 | 0.05445 | 0.06359 | 0.05397 | 0.05501 |
| 120 | 0.15842 | 0.13003 |  | 0.13230 | 0.09639 | 0.12342 | 0.09656 | 0.12116 | 0.09744 | 0.09603 |
| Model | $H$ | Base | original | mse | ✓ |  |  | ✓ | ✓ |  | ✓ |
| dir |  | ✓ |  | ✓ |  | ✓ | ✓ |
| price |  |  | ✓ |  | ✓ | ✓ | ✓ |
| iTransformer | 60 | 0.05752 | 0.04065 |  | 0.04152 | 0.03367 | 0.04882 | 0.03360 | 0.04129 | 0.03329 | 0.03114 |
| 90 | 0.12469 | 0.09315 |  | 0.09210 | 0.06737 | 0.09673 | 0.06748 | 0.08499 | 0.06637 | 0.05752 |
| 120 | 0.22275 | 0.16534 |  | 0.16415 | 0.12091 | 0.18026 | 0.12111 | 0.15351 | 0.11866 | 0.10034 |
| DLinear | 60 | 0.03478 | 0.03641 |  | 0.03594 | 0.02890 | 0.03388 | 0.02885 | 0.03488 | 0.02873 | 0.02851 |
| 90 | 0.07097 | 0.06912 |  | 0.06945 | 0.05287 | 0.06320 | 0.05272 | 0.06362 | 0.05144 | 0.04959 |
| 120 | 0.11325 | 0.10974 |  | 0.11371 | 0.08360 | 0.10743 | 0.08359 | 0.10638 | 0.08270 | 0.07792 |
| FreTS | 60 | 0.02635 | 0.02813 |  | 0.02665 | 0.02516 | 0.02580 | 0.02517 | 0.02647 | 0.02520 | 0.02558 |
| 90 | 0.04852 | 0.05346 |  | 0.04894 | 0.04432 | 0.04718 | 0.04436 | 0.04837 | 0.04446 | 0.04428 |
| 120 | 0.07669 | 0.08457 |  | 0.07767 | 0.06790 | 0.07542 | 0.06802 | 0.07704 | 0.06836 | 0.06736 |
| PatchTST | 60 | 0.05240 | 0.03417 |  | 0.03336 | 0.03156 | 0.04403 | 0.03151 | 0.03366 | 0.03139 | 0.02936 |
| 90 | 0.10923 | 0.06663 |  | 0.06762 | 0.06028 | 0.08248 | 0.06023 | 0.06484 | 0.05988 | 0.05584 |
| 120 | 0.19164 | 0.11587 |  | 0.12027 | 0.10330 | 0.15357 | 0.10339 | 0.11577 | 0.10337 | 0.08818 |
| OLS | 60 | 0.05930 | 0.04433 |  | 0.03844 | 0.03375 | 0.04456 | 0.03383 | 0.03932 | 0.03367 | 0.03216 |
| 90 | 0.12777 | 0.08605 |  | 0.07723 | 0.06642 | 0.08503 | 0.06651 | 0.08104 | 0.06617 | 0.06076 |
| 120 | 0.22788 | 0.14938 |  | 0.14265 | 0.11526 | 0.15923 | 0.11557 | 0.14362 | 0.11667 | 0.10120 |

Table: A.1.1 PETSA Macro and Market Series

| Model | $H$ | Base | original | mse | ✓ |  |  | ✓ | ✓ |  | ✓ |
| dir |  | ✓ |  | ✓ |  | ✓ | ✓ |
| price |  |  | ✓ |  | ✓ | ✓ | ✓ |
| iTransformer | 60 | 0.01886 | 0.01564 |  | 0.01422 | 0.01209 | 0.01602 | 0.01211 | 0.01361 | 0.01196 | 0.01157 |
| 90 | 0.04027 | 0.02911 |  | 0.02804 | 0.02405 | 0.03165 | 0.02404 | 0.02737 | 0.02279 | 0.02004 |
| 120 | 0.07258 | 0.04806 |  | 0.04782 | 0.04021 | 0.05915 | 0.04006 | 0.04824 | 0.03831 | 0.03643 |
| DLinear | 60 | 0.01248 | 0.01361 |  | 0.01309 | 0.01091 | 0.01249 | 0.01085 | 0.01268 | 0.01082 | 0.01064 |
| 90 | 0.02567 | 0.02491 |  | 0.02587 | 0.02114 | 0.02362 | 0.02111 | 0.02295 | 0.02087 | 0.02017 |
| 120 | 0.04046 | 0.04127 |  | 0.04111 | 0.03303 | 0.04040 | 0.03289 | 0.04087 | 0.03266 | 0.03058 |
| FreTS | 60 | 0.00882 | 0.00933 |  | 0.00901 | 0.00909 | 0.00879 | 0.00907 | 0.00899 | 0.00907 | 0.00928 |
| 90 | 0.01670 | 0.01752 |  | 0.01694 | 0.01716 | 0.01676 | 0.01711 | 0.01698 | 0.01708 | 0.01722 |
| 120 | 0.02586 | 0.02707 |  | 0.02628 | 0.02615 | 0.02627 | 0.02612 | 0.02649 | 0.02615 | 0.02624 |
| PatchTST | 60 | 0.01793 | 0.01152 |  | 0.01171 | 0.01176 | 0.01504 | 0.01168 | 0.01155 | 0.01164 | 0.01081 |
| 90 | 0.03668 | 0.02402 |  | 0.02521 | 0.02488 | 0.02933 | 0.02467 | 0.02417 | 0.02389 | 0.02053 |
| 120 | 0.06088 | 0.04170 |  | 0.04323 | 0.03983 | 0.05006 | 0.03955 | 0.03967 | 0.03866 | 0.03350 |
| OLS | 60 | 0.02018 | 0.01708 |  | 0.01359 | 0.01213 | 0.01477 | 0.01204 | 0.01307 | 0.01216 | 0.01112 |
| 90 | 0.04483 | 0.03102 |  | 0.02809 | 0.02424 | 0.02880 | 0.02420 | 0.02505 | 0.02363 | 0.02134 |
| 120 | 0.07908 | 0.05202 |  | 0.04628 | 0.04014 | 0.05234 | 0.04018 | 0.04400 | 0.03965 | 0.03511 |

#### A.2 Additional Quantitative Results

##### A.2.1 Relative Improvements in Reconstructed Price MSE by Dataset

To compare changes in reconstructed-price error after replacing the original adaptation objectives with DPAL, Tables A.2.2 and A.2.2 report the relative improvements obtained by replacing the original adaptation objectives of TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) with DPAL, respectively. The reported percentages are calculated from the MSE values shown in Tables 1 and 2 .

Table: Table A.2.1 : TAFAS Relative Improvement

| Datasets | H | iTransformer | DLinear | FreTS | PatchTST | OLS |
| Magnificent 7 | 60 | 51.9% | 35.7% | 9.8% | 36.0% | 53.9% |
| 90 | 59.2% | 52.7% | 19.0% | 50.3% | 51.9% |
| 120 | 69.4% | 59.2% | 20.5% | 62.3% | 55.8% |
| S&P 500 Excerpt | 60 | 55.9% | 31.7% | 5.1% | 31.9% | 51.9% |
| 90 | 62.3% | 44.0% | 9.9% | 38.2% | 55.5% |
| 120 | 64.5% | 51.3% | 15.4% | 32.1% | 60.6% |
| Russell 2000 Excerpt | 60 | 52.9% | 42.1% | 4.3% | 29.0% | 45.6% |
| 90 | 62.5% | 50.2% | 11.2% | 41.6% | 57.4% |
| 120 | 63.1% | 46.9% | 8.1% | 47.9% | 60.3% |
| Technology Sector | 60 | 52.3% | 36.6% | 9.2% | 17.4% | 48.0% |
| 90 | 63.7% | 49.1% | 14.5% | 32.5% | 50.4% |
| 120 | 76.2% | 52.6% | 16.1% | 49.4% | 68.3% |
| Consumer Sector | 60 | 43.5% | 21.4% | 0.0% | 25.0% | 40.9% |
| 90 | 49.0% | 25.0% | 5.6% | 25.0% | 44.9% |
| 120 | 52.0% | 26.1% | 6.7% | 25.9% | 45.7% |
| Energy Sector | 60 | 33.3% | 14.8% | 0.0% | 23.3% | 31.4% |
| 90 | 39.2% | 18.5% | -2.9% | 23.3% | 35.5% |
| 120 | 41.1% | 21.5% | -1.7% | 22.7% | 36.4% |
| Nasdaq Mixed Excerpt | 60 | 39.2% | 20.0% | 0.0% | 19.4% | 28.9% |
| 90 | 46.4% | 27.1% | 10.2% | 25.4% | 33.0% |
| 120 | 47.2% | 29.5% | 11.7% | 27.4% | 36.7% |
| Macro and Market Series | 60 | 31.3% | 8.3% | 0.0% | 16.7% | 26.7% |
| 90 | 40.0% | 20.0% | 0.0% | 20.8% | 33.3% |
| 120 | 41.0% | 22.5% | 0.0% | 22.5% | 38.6% |
| Datasets | H | iTransformer | DLinear | FreTS | PatchTST | OLS |
| Magnificent 7 | 60 | 27.5% | 34.4% | 18.9% | 31.0% | 41.6% |
| 90 | 46.6% | 44.5% | 25.9% | 43.8% | 47.1% |
| 120 | 48.6% | 56.9% | 31.7% | 49.8% | 48.0% |
| S&P 500 Excerpt | 60 | 29.1% | 41.7% | 13.8% | 4.6% | 30.6% |
| 90 | 26.3% | 49.1% | 17.7% | 5.2% | 22.6% |
| 120 | 27.2% | 57.6% | 25.5% | 15.0% | 25.3% |
| Russell 2000 Excerpt | 60 | 38.5% | 45.7% | 19.8% | 20.8% | 41.5% |
| 90 | 75.2% | 43.2% | 24.4% | 32.6% | 44.3% |
| 120 | 52.4% | 40.9% | 22.1% | 33.8% | 47.6% |
| Technology Sector | 60 | 49.5% | 52.8% | 21.6% | 27.0% | 50.2% |
| 90 | 58.0% | 38.6% | 29.2% | 35.1% | 60.6% |
| 120 | 63.6% | 55.6% | 37.8% | 24.5% | 67.0% |
| Consumer Sector | 60 | 35.0% | 31.3% | 16.7% | 25.0% | 38.1% |
| 90 | 42.2% | 32.3% | 22.7% | 29.0% | 39.5% |
| 120 | 46.0% | 32.7% | 25.7% | 29.6% | 41.0% |
| Energy Sector | 60 | 24.2% | 17.9% | 5.0% | 6.9% | 18.2% |
| 90 | 23.2% | 25.9% | 5.3% | 7.1% | 22.5% |
| 120 | 26.0% | 25.0% | 9.2% | 11.0% | 26.2% |
| Nasdaq Mixed Excerpt | 60 | 24.4% | 19.4% | 7.1% | 14.7% | 27.3% |
| 90 | 37.6% | 27.5% | 17.0% | 16.4% | 29.1% |
| 120 | 39.4% | 29.1% | 21.2% | 24.1% | 32.2% |
| Macro and Market Series | 60 | 25.0% | 21.4% | 0.0% | 8.3% | 35.3% |
| 90 | 31.0% | 20.0% | 5.6% | 12.5% | 32.3% |
| 120 | 25.0% | 24.4% | 3.7% | 19.0% | 32.7% |

For an adaptation framework ${f}\in\{\mathrm{TAFAS},\mathrm{PETSA}\}$ , the relative improvement is defined as

$$I_{f}=\frac{\operatorname{MSE}_{f}-\operatorname{MSE}_{f\text{+}\mathrm{DPAL}}}{\operatorname{MSE}_{f}}\times 100\%.$$

A positive value indicates that the setting using DPAL has a lower MSE of reconstructed prices than the corresponding baseline framework, whereas a negative value indicates a higher error.

For each framework, positive relative improvements occur in most combinations of datasets, forecasting backbones, and forecast horizons. The magnitude of the improvement varies across datasets, forecasting backbones, and forecast horizons.

Relative improvements are calculated from the MSE values rounded to three decimal places in Tables 1 and 2 . Percentages close to zero may be affected by rounding.

##### A.2.2 Pointwise Errors in Standardized Return Space

The primary evaluation in the main results measures the MSE of reconstructed relative prices. For completeness, Tables A.2.3 – A.2.6 additionally report pointwise MSE and MAE computed directly in the standardized return space. These MSE and MAE are calculated before inverse transformation and price reconstruction. Lower values indicate better pointwise return accuracy. All comparison counts in this subsection are calculated using the unrounded results, whereas the values displayed in the tables are rounded to three decimal places.

Table: Table A.2.3 : Forecasting performance of TAFAS measured by MSE in the standardized return space. Lower values are better. The best result within each backbone and horizon is highlighted in green.

| Datasets | $H$ | iTransformer | DLinear | FreTS | PatchTST | OLS |
| Base | TAFAS | with DPAL | Base | TAFAS | with DPAL | Base | TAFAS | with DPAL | Base | TAFAS | with DPAL | Base | TAFAS | with DPAL |
| Magnificent 7 | 60 | 1.150 | 1.139 | 1.126 | 1.114 | 1.113 | 1.110 | 1.135 | 1.134 | 1.125 | 1.145 | 1.129 | 1.128 | 1.139 | 1.127 | 1.120 |
| 90 | 1.152 | 1.141 | 1.135 | 1.123 | 1.122 | 1.119 | 1.137 | 1.136 | 1.129 | 1.154 | 1.136 | 1.139 | 1.146 | 1.133 | 1.130 |
| 120 | 1.162 | 1.152 | 1.144 | 1.133 | 1.132 | 1.130 | 1.144 | 1.143 | 1.138 | 1.160 | 1.144 | 1.144 | 1.155 | 1.144 | 1.140 |
| S&P 500 Excerpt | 60 | 1.513 | 1.508 | 1.488 | 1.476 | 1.476 | 1.472 | 1.502 | 1.501 | 1.489 | 1.507 | 1.494 | 1.486 | 1.500 | 1.494 | 1.482 |
| 90 | 1.497 | 1.493 | 1.476 | 1.471 | 1.470 | 1.466 | 1.485 | 1.485 | 1.477 | 1.502 | 1.491 | 1.483 | 1.491 | 1.486 | 1.475 |
| 120 | 1.496 | 1.493 | 1.476 | 1.469 | 1.469 | 1.464 | 1.480 | 1.480 | 1.474 | 1.495 | 1.485 | 1.479 | 1.489 | 1.484 | 1.475 |
| Russell 2000 Excerpt | 60 | 1.116 | 1.110 | 1.092 | 1.081 | 1.080 | 1.075 | 1.100 | 1.099 | 1.088 | 1.102 | 1.093 | 1.086 | 1.103 | 1.096 | 1.088 |
| 90 | 1.118 | 1.112 | 1.097 | 1.092 | 1.091 | 1.085 | 1.103 | 1.103 | 1.095 | 1.110 | 1.101 | 1.096 | 1.110 | 1.103 | 1.096 |
| 120 | 1.120 | 1.116 | 1.103 | 1.098 | 1.098 | 1.093 | 1.105 | 1.105 | 1.099 | 1.116 | 1.107 | 1.103 | 1.116 | 1.110 | 1.103 |
| Technology Sector | 60 | 1.465 | 1.457 | 1.440 | 1.433 | 1.432 | 1.429 | 1.451 | 1.450 | 1.441 | 1.459 | 1.444 | 1.440 | 1.452 | 1.444 | 1.437 |
| 90 | 1.458 | 1.450 | 1.439 | 1.436 | 1.435 | 1.431 | 1.444 | 1.444 | 1.438 | 1.457 | 1.446 | 1.442 | 1.452 | 1.443 | 1.438 |
| 120 | 1.458 | 1.453 | 1.441 | 1.436 | 1.435 | 1.432 | 1.442 | 1.442 | 1.437 | 1.453 | 1.444 | 1.439 | 1.451 | 1.445 | 1.439 |
| Consumer Sector | 60 | 1.219 | 1.214 | 1.196 | 1.192 | 1.191 | 1.186 | 1.205 | 1.205 | 1.195 | 1.213 | 1.205 | 1.197 | 1.208 | 1.203 | 1.192 |
| 90 | 1.223 | 1.219 | 1.207 | 1.203 | 1.202 | 1.197 | 1.210 | 1.210 | 1.203 | 1.222 | 1.215 | 1.211 | 1.218 | 1.213 | 1.206 |
| 120 | 1.241 | 1.238 | 1.228 | 1.220 | 1.219 | 1.215 | 1.226 | 1.226 | 1.219 | 1.237 | 1.231 | 1.228 | 1.236 | 1.231 | 1.226 |
| Energy Sector | 60 | 0.576 | 0.574 | 0.562 | 0.557 | 0.555 | 0.550 | 0.567 | 0.566 | 0.560 | 0.571 | 0.565 | 0.560 | 0.565 | 0.563 | 0.559 |
| 90 | 0.573 | 0.571 | 0.562 | 0.558 | 0.557 | 0.551 | 0.566 | 0.566 | 0.561 | 0.572 | 0.566 | 0.562 | 0.567 | 0.565 | 0.561 |
| 120 | 0.578 | 0.577 | 0.570 | 0.562 | 0.562 | 0.557 | 0.569 | 0.569 | 0.565 | 0.574 | 0.570 | 0.567 | 0.572 | 0.571 | 0.567 |
| Nasdaq Mixed Excerpt | 60 | 0.969 | 0.964 | 0.947 | 0.947 | 0.946 | 0.940 | 0.958 | 0.957 | 0.947 | 0.963 | 0.956 | 0.948 | 0.961 | 0.957 | 0.947 |
| 90 | 0.963 | 0.960 | 0.947 | 0.948 | 0.947 | 0.942 | 0.953 | 0.953 | 0.946 | 0.962 | 0.956 | 0.949 | 0.960 | 0.956 | 0.949 |
| 120 | 0.965 | 0.962 | 0.952 | 0.950 | 0.949 | 0.945 | 0.953 | 0.953 | 0.948 | 0.962 | 0.957 | 0.952 | 0.962 | 0.959 | 0.953 |
| Macro and Market Series | 60 | 0.792 | 0.786 | 0.774 | 0.769 | 0.767 | 0.765 | 0.787 | 0.786 | 0.776 | 0.788 | 0.780 | 0.774 | 0.786 | 0.779 | 0.774 |
| 90 | 0.793 | 0.788 | 0.778 | 0.777 | 0.774 | 0.772 | 0.788 | 0.787 | 0.780 | 0.796 | 0.789 | 0.784 | 0.790 | 0.784 | 0.780 |
| 120 | 0.803 | 0.799 | 0.791 | 0.788 | 0.786 | 0.784 | 0.797 | 0.791 | 0.791 | 0.805 | 0.799 | 0.796 | 0.800 | 0.794 | 0.791 |

Table: Table A.2.4 : Forecasting performance of TAFAS measured by MAE in the standardized return space. Lower values are better. The best result within each backbone and horizon is highlighted in green.

| Datasets | $H$ | iTransformer | DLinear | FreTS | PatchTST | OLS |
| Base | TAFAS | with DPAL | Base | TAFAS | with DPAL | Base | TAFAS | with DPAL | Base | TAFAS | with DPAL | Base | TAFAS | with DPAL |
| Magnificent 7 | 60 | 0.761 | 0.755 | 0.750 | 0.742 | 0.742 | 0.741 | 0.754 | 0.753 | 0.748 | 0.758 | 0.751 | 0.750 | 0.756 | 0.750 | 0.747 |
| 90 | 0.759 | 0.754 | 0.752 | 0.745 | 0.744 | 0.742 | 0.752 | 0.752 | 0.748 | 0.760 | 0.752 | 0.753 | 0.756 | 0.750 | 0.749 |
| 120 | 0.761 | 0.757 | 0.754 | 0.747 | 0.746 | 0.746 | 0.753 | 0.753 | 0.750 | 0.760 | 0.753 | 0.753 | 0.757 | 0.753 | 0.751 |
| S&P 500 Excerpt | 60 | 0.822 | 0.820 | 0.811 | 0.806 | 0.806 | 0.805 | 0.817 | 0.817 | 0.812 | 0.820 | 0.814 | 0.811 | 0.817 | 0.814 | 0.809 |
| 90 | 0.816 | 0.814 | 0.807 | 0.805 | 0.805 | 0.803 | 0.811 | 0.811 | 0.808 | 0.818 | 0.813 | 0.810 | 0.813 | 0.811 | 0.807 |
| 120 | 0.816 | 0.816 | 0.809 | 0.806 | 0.806 | 0.804 | 0.811 | 0.810 | 0.808 | 0.816 | 0.812 | 0.810 | 0.814 | 0.812 | 0.809 |
| Russell 2000 Excerpt | 60 | 0.714 | 0.712 | 0.701 | 0.696 | 0.695 | 0.692 | 0.705 | 0.705 | 0.699 | 0.706 | 0.701 | 0.698 | 0.707 | 0.704 | 0.699 |
| 90 | 0.711 | 0.709 | 0.701 | 0.698 | 0.697 | 0.694 | 0.704 | 0.704 | 0.700 | 0.707 | 0.703 | 0.700 | 0.707 | 0.704 | 0.700 |
| 120 | 0.710 | 0.708 | 0.702 | 0.699 | 0.699 | 0.696 | 0.703 | 0.703 | 0.700 | 0.707 | 0.704 | 0.702 | 0.707 | 0.705 | 0.702 |
| Technology Sector | 60 | 0.792 | 0.788 | 0.780 | 0.777 | 0.777 | 0.775 | 0.785 | 0.785 | 0.781 | 0.790 | 0.782 | 0.780 | 0.786 | 0.782 | 0.779 |
| 90 | 0.787 | 0.783 | 0.778 | 0.777 | 0.776 | 0.774 | 0.781 | 0.781 | 0.778 | 0.786 | 0.782 | 0.779 | 0.784 | 0.780 | 0.778 |
| 120 | 0.787 | 0.785 | 0.780 | 0.777 | 0.777 | 0.775 | 0.780 | 0.780 | 0.778 | 0.785 | 0.781 | 0.779 | 0.784 | 0.781 | 0.779 |
| Consumer Sector | 60 | 0.759 | 0.757 | 0.749 | 0.748 | 0.748 | 0.746 | 0.754 | 0.753 | 0.749 | 0.757 | 0.754 | 0.750 | 0.755 | 0.753 | 0.748 |
| 90 | 0.758 | 0.756 | 0.751 | 0.751 | 0.750 | 0.748 | 0.753 | 0.753 | 0.750 | 0.758 | 0.755 | 0.753 | 0.756 | 0.755 | 0.751 |
| 120 | 0.761 | 0.760 | 0.756 | 0.753 | 0.753 | 0.751 | 0.756 | 0.756 | 0.753 | 0.760 | 0.758 | 0.757 | 0.759 | 0.758 | 0.756 |
| Energy Sector | 60 | 0.541 | 0.540 | 0.532 | 0.528 | 0.527 | 0.523 | 0.535 | 0.535 | 0.531 | 0.538 | 0.534 | 0.532 | 0.534 | 0.533 | 0.530 |
| 90 | 0.537 | 0.536 | 0.530 | 0.526 | 0.525 | 0.522 | 0.532 | 0.532 | 0.529 | 0.537 | 0.533 | 0.530 | 0.532 | 0.532 | 0.529 |
| 120 | 0.537 | 0.537 | 0.532 | 0.527 | 0.526 | 0.523 | 0.531 | 0.531 | 0.529 | 0.535 | 0.532 | 0.530 | 0.533 | 0.532 | 0.530 |
| Nasdaq Mixed Excerpt | 60 | 0.682 | 0.679 | 0.671 | 0.671 | 0.670 | 0.668 | 0.677 | 0.676 | 0.672 | 0.679 | 0.675 | 0.672 | 0.678 | 0.675 | 0.672 |
| 90 | 0.678 | 0.676 | 0.671 | 0.671 | 0.670 | 0.668 | 0.674 | 0.673 | 0.671 | 0.678 | 0.675 | 0.672 | 0.677 | 0.675 | 0.672 |
| 120 | 0.678 | 0.677 | 0.673 | 0.671 | 0.671 | 0.669 | 0.673 | 0.673 | 0.671 | 0.677 | 0.675 | 0.673 | 0.677 | 0.675 | 0.673 |
| Macro and Market Series | 60 | 0.616 | 0.613 | 0.608 | 0.605 | 0.604 | 0.603 | 0.616 | 0.615 | 0.609 | 0.614 | 0.611 | 0.609 | 0.612 | 0.610 | 0.608 |
| 90 | 0.614 | 0.612 | 0.608 | 0.607 | 0.605 | 0.604 | 0.613 | 0.613 | 0.609 | 0.616 | 0.613 | 0.612 | 0.612 | 0.610 | 0.609 |
| 120 | 0.617 | 0.615 | 0.613 | 0.610 | 0.609 | 0.608 | 0.615 | 0.614 | 0.612 | 0.619 | 0.616 | 0.615 | 0.613 | 0.613 | 0.613 |

Table: Table A.2.5 : Forecasting performance of PETSA measured by MSE in the standardized return space. Lower values are better. The best result within each backbone and horizon is highlighted in green.

| Datasets | $H$ | iTransformer | DLinear | FreTS | PatchTST | OLS |
| Base | PETSA | with DPAL | Base | PETSA | with DPAL | Base | PETSA | with DPAL | Base | PETSA | with DPAL | Base | PETSA | with DPAL |
| Magnificent 7 | 60 | 1.150 | 1.144 | 1.143 | 1.114 | 1.123 | 1.112 | 1.135 | 1.136 | 1.132 | 1.145 | 1.136 | 1.138 | 1.139 | 1.137 | 1.129 |
| 90 | 1.152 | 1.144 | 1.138 | 1.123 | 1.123 | 1.121 | 1.137 | 1.137 | 1.135 | 1.154 | 1.142 | 1.141 | 1.146 | 1.137 | 1.135 |
| 120 | 1.162 | 1.153 | 1.145 | 1.133 | 1.133 | 1.131 | 1.144 | 1.145 | 1.142 | 1.160 | 1.150 | 1.145 | 1.155 | 1.147 | 1.144 |
| S&P 500 Excerpt | 60 | 1.513 | 1.515 | 1.490 | 1.476 | 1.477 | 1.472 | 1.502 | 1.502 | 1.494 | 1.507 | 1.503 | 1.489 | 1.500 | 1.502 | 1.485 |
| 90 | 1.497 | 1.492 | 1.478 | 1.471 | 1.472 | 1.466 | 1.485 | 1.485 | 1.481 | 1.502 | 1.493 | 1.488 | 1.491 | 1.486 | 1.477 |
| 120 | 1.496 | 1.490 | 1.479 | 1.469 | 1.470 | 1.465 | 1.480 | 1.480 | 1.477 | 1.495 | 1.485 | 1.482 | 1.489 | 1.483 | 1.476 |
| Russell 2000 Excerpt | 60 | 1.116 | 1.113 | 1.100 | 1.081 | 1.083 | 1.076 | 1.100 | 1.100 | 1.093 | 1.102 | 1.095 | 1.095 | 1.103 | 1.102 | 1.093 |
| 90 | 1.118 | 1.114 | 1.100 | 1.092 | 1.092 | 1.086 | 1.103 | 1.104 | 1.098 | 1.110 | 1.103 | 1.102 | 1.110 | 1.111 | 1.100 |
| 120 | 1.120 | 1.118 | 1.104 | 1.098 | 1.098 | 1.093 | 1.105 | 1.106 | 1.102 | 1.116 | 1.110 | 1.106 | 1.116 | 1.113 | 1.105 |
| Technology Sector | 60 | 1.465 | 1.461 | 1.445 | 1.433 | 1.436 | 1.430 | 1.451 | 1.453 | 1.446 | 1.459 | 1.453 | 1.442 | 1.452 | 1.454 | 1.440 |
| 90 | 1.458 | 1.454 | 1.444 | 1.436 | 1.438 | 1.432 | 1.444 | 1.445 | 1.442 | 1.457 | 1.448 | 1.446 | 1.452 | 1.451 | 1.441 |
| 120 | 1.458 | 1.456 | 1.444 | 1.436 | 1.438 | 1.432 | 1.442 | 1.443 | 1.440 | 1.453 | 1.447 | 1.443 | 1.451 | 1.448 | 1.441 |
| Consumer Sector | 60 | 1.219 | 1.216 | 1.199 | 1.192 | 1.193 | 1.186 | 1.205 | 1.206 | 1.199 | 1.213 | 1.208 | 1.201 | 1.208 | 1.208 | 1.195 |
| 90 | 1.223 | 1.220 | 1.210 | 1.203 | 1.204 | 1.198 | 1.210 | 1.211 | 1.206 | 1.222 | 1.217 | 1.213 | 1.218 | 1.218 | 1.210 |
| 120 | 1.241 | 1.241 | 1.230 | 1.220 | 1.220 | 1.216 | 1.226 | 1.226 | 1.222 | 1.237 | 1.233 | 1.229 | 1.236 | 1.235 | 1.231 |
| Energy Sector | 60 | 0.576 | 0.575 | 0.563 | 0.557 | 0.558 | 0.551 | 0.567 | 0.568 | 0.564 | 0.571 | 0.567 | 0.564 | 0.565 | 0.565 | 0.562 |
| 90 | 0.573 | 0.571 | 0.562 | 0.558 | 0.558 | 0.551 | 0.566 | 0.567 | 0.564 | 0.572 | 0.571 | 0.564 | 0.567 | 0.567 | 0.564 |
| 120 | 0.578 | 0.577 | 0.570 | 0.562 | 0.563 | 0.557 | 0.569 | 0.569 | 0.567 | 0.574 | 0.574 | 0.568 | 0.572 | 0.572 | 0.569 |
| Nasdaq Mixed Excerpt | 60 | 0.969 | 0.967 | 0.950 | 0.947 | 0.950 | 0.942 | 0.957 | 0.959 | 0.951 | 0.963 | 0.959 | 0.951 | 0.961 | 0.966 | 0.951 |
| 90 | 0.963 | 0.961 | 0.950 | 0.948 | 0.949 | 0.943 | 0.953 | 0.954 | 0.948 | 0.962 | 0.957 | 0.953 | 0.960 | 0.965 | 0.952 |
| 120 | 0.965 | 0.963 | 0.954 | 0.950 | 0.950 | 0.945 | 0.953 | 0.954 | 0.949 | 0.962 | 0.958 | 0.954 | 0.962 | 0.961 | 0.955 |
| Macro and Market Series | 60 | 0.792 | 0.790 | 0.779 | 0.769 | 0.769 | 0.765 | 0.787 | 0.787 | 0.782 | 0.788 | 0.784 | 0.777 | 0.786 | 0.788 | 0.780 |
| 90 | 0.793 | 0.790 | 0.781 | 0.777 | 0.775 | 0.771 | 0.788 | 0.787 | 0.784 | 0.796 | 0.792 | 0.787 | 0.790 | 0.786 | 0.783 |
| 120 | 0.803 | 0.801 | 0.792 | 0.788 | 0.787 | 0.783 | 0.796 | 0.796 | 0.793 | 0.805 | 0.801 | 0.799 | 0.800 | 0.796 | 0.794 |

Table: Table A.2.6 : Forecasting performance of PETSA measured by MAE in the standardized return space. Lower values are better. The best result within each backbone and horizon is highlighted in green.

| Datasets | $H$ | iTransformer | DLinear | FreTS | PatchTST | OLS |
| Base | PETSA | with DPAL | Base | PETSA | with DPAL | Base | PETSA | with DPAL | Base | PETSA | with DPAL | Base | PETSA | with DPAL |
| Magnificent 7 | 60 | 0.761 | 0.759 | 0.757 | 0.742 | 0.745 | 0.742 | 0.754 | 0.753 | 0.751 | 0.758 | 0.754 | 0.755 | 0.756 | 0.754 | 0.751 |
| 90 | 0.759 | 0.756 | 0.752 | 0.745 | 0.744 | 0.743 | 0.752 | 0.752 | 0.751 | 0.760 | 0.755 | 0.754 | 0.756 | 0.751 | 0.752 |
| 120 | 0.761 | 0.757 | 0.754 | 0.747 | 0.747 | 0.746 | 0.753 | 0.754 | 0.752 | 0.760 | 0.755 | 0.753 | 0.757 | 0.754 | 0.754 |
| S&P 500 Excerpt | 60 | 0.822 | 0.822 | 0.813 | 0.806 | 0.806 | 0.805 | 0.817 | 0.817 | 0.814 | 0.820 | 0.817 | 0.812 | 0.817 | 0.817 | 0.811 |
| 90 | 0.816 | 0.814 | 0.808 | 0.805 | 0.805 | 0.803 | 0.811 | 0.811 | 0.810 | 0.818 | 0.814 | 0.813 | 0.813 | 0.811 | 0.808 |
| 120 | 0.816 | 0.815 | 0.811 | 0.806 | 0.806 | 0.805 | 0.811 | 0.810 | 0.810 | 0.816 | 0.812 | 0.812 | 0.814 | 0.812 | 0.809 |
| Russell 2000 Excerpt | 60 | 0.714 | 0.712 | 0.706 | 0.696 | 0.697 | 0.693 | 0.705 | 0.705 | 0.702 | 0.706 | 0.702 | 0.703 | 0.707 | 0.706 | 0.702 |
| 90 | 0.711 | 0.710 | 0.703 | 0.698 | 0.698 | 0.695 | 0.704 | 0.704 | 0.701 | 0.707 | 0.703 | 0.703 | 0.707 | 0.707 | 0.702 |
| 120 | 0.710 | 0.709 | 0.703 | 0.699 | 0.699 | 0.696 | 0.703 | 0.703 | 0.701 | 0.707 | 0.705 | 0.703 | 0.707 | 0.707 | 0.703 |
| Technology Sector | 60 | 0.792 | 0.790 | 0.782 | 0.777 | 0.778 | 0.776 | 0.785 | 0.786 | 0.783 | 0.790 | 0.786 | 0.781 | 0.786 | 0.787 | 0.781 |
| 90 | 0.787 | 0.786 | 0.780 | 0.777 | 0.777 | 0.775 | 0.781 | 0.781 | 0.780 | 0.786 | 0.783 | 0.782 | 0.784 | 0.783 | 0.779 |
| 120 | 0.787 | 0.787 | 0.781 | 0.777 | 0.777 | 0.776 | 0.780 | 0.781 | 0.779 | 0.785 | 0.783 | 0.781 | 0.784 | 0.783 | 0.780 |
| Consumer Sector | 60 | 0.759 | 0.757 | 0.751 | 0.748 | 0.748 | 0.746 | 0.754 | 0.754 | 0.751 | 0.757 | 0.755 | 0.752 | 0.755 | 0.755 | 0.750 |
| 90 | 0.758 | 0.757 | 0.753 | 0.751 | 0.750 | 0.748 | 0.753 | 0.754 | 0.751 | 0.758 | 0.756 | 0.754 | 0.756 | 0.756 | 0.753 |
| 120 | 0.761 | 0.761 | 0.757 | 0.753 | 0.753 | 0.751 | 0.756 | 0.756 | 0.754 | 0.760 | 0.759 | 0.757 | 0.759 | 0.759 | 0.757 |
| Energy Sector | 60 | 0.541 | 0.540 | 0.533 | 0.528 | 0.528 | 0.524 | 0.535 | 0.536 | 0.533 | 0.538 | 0.535 | 0.534 | 0.534 | 0.535 | 0.532 |
| 90 | 0.537 | 0.536 | 0.530 | 0.526 | 0.526 | 0.522 | 0.532 | 0.532 | 0.531 | 0.537 | 0.535 | 0.532 | 0.532 | 0.532 | 0.531 |
| 120 | 0.537 | 0.536 | 0.533 | 0.527 | 0.527 | 0.523 | 0.531 | 0.531 | 0.530 | 0.535 | 0.534 | 0.531 | 0.533 | 0.533 | 0.531 |
| Nasdaq Mixed Excerpt | 60 | 0.682 | 0.680 | 0.673 | 0.671 | 0.672 | 0.669 | 0.676 | 0.677 | 0.674 | 0.679 | 0.677 | 0.674 | 0.678 | 0.679 | 0.674 |
| 90 | 0.678 | 0.677 | 0.672 | 0.671 | 0.671 | 0.669 | 0.674 | 0.674 | 0.672 | 0.678 | 0.675 | 0.674 | 0.677 | 0.678 | 0.674 |
| 120 | 0.678 | 0.677 | 0.674 | 0.671 | 0.671 | 0.669 | 0.673 | 0.673 | 0.672 | 0.677 | 0.675 | 0.674 | 0.677 | 0.676 | 0.674 |
| Macro and Market Series | 60 | 0.616 | 0.616 | 0.612 | 0.605 | 0.606 | 0.603 | 0.615 | 0.615 | 0.613 | 0.614 | 0.613 | 0.610 | 0.612 | 0.615 | 0.612 |
| 90 | 0.614 | 0.614 | 0.610 | 0.607 | 0.606 | 0.604 | 0.613 | 0.613 | 0.612 | 0.616 | 0.615 | 0.613 | 0.612 | 0.612 | 0.612 |
| 120 | 0.617 | 0.617 | 0.614 | 0.610 | 0.610 | 0.608 | 0.615 | 0.614 | 0.612 | 0.619 | 0.617 | 0.617 | 0.615 | 0.614 | 0.615 |

Based on the unrounded results underlying the four tables, DPAL achieves lower standardized return errors than the corresponding original adaptation objective in 470 of the 480 comparisons. Specifically, it reduces MSE and MAE in 119 and 118 of the 120 TAFAS ( Kim et al., 2025 ) settings, respectively, and in 118 and 115 of the 120 PETSA ( Medeiros et al., 2025 ) settings. The ten exceptions are minor, with the largest relative increase being approximately $0.24\%$ . Compared with the Base model, DPAL achieves lower errors in almost all settings.

Although the reductions in standardized return MSE and MAE are generally less pronounced than those in reconstructed price MSE, they are observed across nearly all evaluated settings. This difference in magnitude is consistent with the cumulative nature of price reconstruction, as small improvements in pointwise return forecasts can become more visible when returns are compounded over the forecast horizon. Thus, the benefits of DPAL extend beyond reconstructed price accuracy to the pointwise accuracy of standardized return forecasts.

##### A.2.3 Scale-consistent Evaluation

As described in Section 4.1.4, DPAL computes its directional and price calibration losses using training-set inverse standardization, whereas the main reconstructed-price evaluation uses history-conditioned rescaling. To assess whether the gains persist when these transformations are aligned, we additionally evaluate both Original TTA and TTA + DPAL by replacing Eq. (8) with

$$\hat{r}^{sc}_{w,h,c}=\hat{r}^{norm}_{w,h,c}\sigma^{sc}_{c}+\mu^{sc}_{c}.$$

Here, $\mu^{sc}_{c}$ and $\sigma^{sc}_{c}$ are the training-set statistics defined in Section 4.1.4 and remain fixed across evaluation windows. Price reconstruction and MSE computation follow Eqs. (9)–(10), using $\hat{r}^{sc}_{w,h,c}$ as the predicted returns and retaining the observed returns on their original scale. We denote the resulting reconstructed-price error by $\mathrm{MSE}^{sc}_{price}$ . For each framework, we compare the two methods across the same $S=120$ combinations of datasets, forecasting backbones, and forecast horizons. For setting $s$ , we define the MSE ratio and relative improvement as

$$\rho_{s}=\frac{\mathrm{MSE}^{sc}_{DPAL,s}}{\mathrm{MSE}^{sc}_{Original,s}},\qquad I_{s}=(1-\rho_{s})\times 100\%.$$

We report the number of settings in Table A.2.7 with strictly lower MSE under DPAL ( $\rho_{s}<1$ ), the median of $I_{s}$ , and the geometric-mean improvement, defined as

$$\qquad I_{geo}=\left[1-\exp\left(\frac{1}{S}\sum_{s=1}^{S}\log\rho_{s}\right)\right]\times 100\%.$$

Table: Table A.2.7 : Scale-consistent reconstructed-price evaluation using training-set inverse standardization. Improvements are relative to Original TTA; positive values indicate lower MSE with DPAL.

| Framework | Settings improved by DPAL | Median improvement | Geometric-mean improvement |
| TAFAS | 102/120 | 17.24% | 18.24% |
| PETSA | 104/120 | 20.85% | 19.66% |

DPAL reduces reconstructed-price MSE in most settings under both frameworks, with positive median and geometric-mean improvements. These results indicate that the observed gains are not solely attributable to history-conditioned rescaling.

#### A.3 Statistical Analysis

##### A.3.1 Exact Wilcoxon signed-rank test

To assess whether the improvement obtained by DPAL is consistent across datasets, rather than being driven by a large number of backbone–horizon configurations from a small number of datasets, we conducted a dataset-level statistical analysis based on Tables A.1 . For each dataset ( $d$ ), backbone ( $m$ ), and forecasting horizon ( $h$ ), we first calculated the natural logarithm of the ratio between the reconstructed price MSE obtained with DPAL and that obtained with the original adaptation objective:

$$g_{d,m,h}=\log\left(\frac{\mathrm{MSE}_{\mathrm{DPAL},d,m,h}}{\mathrm{MSE}_{\mathrm{Original},d,m,h}}\right).$$ (13)

Here, $\mathrm{MSE}_{\mathrm{Original}}$ denotes the result in the first adaptation column to the right of the frozen Base model, whereas $\mathrm{MSE}_{\mathrm{DPAL}}$ denotes the result in the last column, corresponding to the complete DPAL objective. A negative value of $g_{d,m,h}$ indicates that DPAL achieves a lower reconstructed price MSE than the original objective for the corresponding configuration.

Each table contains five forecasting backbones and three forecasting horizons, $h\in\{60,90,120\}$ , resulting in 15 configurations per dataset. We therefore summarized the configuration-level log ratios within each dataset as

$$K_{d}=\frac{1}{15}\sum_{m=1}^{5}\sum_{h\in\{60,90,120\}}g_{d,m,h}.$$ (14)

Equivalently, $\exp(K_{d})$ represents the geometric mean of the DPAL-to-Original MSE ratios across the 15 configurations. Aggregation was performed before hypothesis testing because configurations evaluated on the same dataset were not treated as independent experimental units.

The resulting dataset-level scores are reported in Table A.3.1 . All eight scores are negative under both TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) .

Table: Table A.3.1 : Dataset-level mean log MSE ratios ( $K_{d}$ ). Negative values indicate lower reconstructed price MSE for DPAL than for the original adaptation objective.

| Dataset | TAFAS | PETSA |
| Magnificent 7 | -0.659118 | -0.521236 |
| S&P 500 Excerpt | -0.571172 | -0.322853 |
| Russell 2000 Excerpt | -0.585539 | -0.524676 |
| Technology Sector | -0.617923 | -0.629437 |
| Consumer Sector | -0.367755 | -0.402655 |
| Energy Sector | -0.270304 | -0.190842 |
| Nasdaq Mixed Excerpt | -0.327553 | -0.289190 |
| Macro and Market Series | -0.252861 | -0.226572 |

We then applied an exact one-sided Wilcoxon signed-rank test separately to the eight dataset-level scores obtained under each adaptation framework. Let $\theta$ denote the location parameter of the dataset-level log ratios. The hypotheses were

$$H_{0}:\theta\geq 0,\qquad H_{1}:\theta<0,$$ (15)

with the exact null distribution evaluated at $\theta=0$ . The significance level was set to $\alpha=0.05$ .

For TAFAS ( Kim et al., 2025 ) , all eight $K_{d}$ values were negative, yielding a positive-rank sum of $W^{+}=0$ and a negative-rank sum of $W^{-}=36$ . The exact one-sided test gave $p=0.00390625$ . The same result was obtained for PETSA ( Medeiros et al., 2025 ) , for which all eight scores were also negative ( $W^{+}=0$ , $W^{-}=36$ , and $p=0.00390625$ ). Because there were no zero differences or tied absolute ranks, the exact probability was obtained from the $2^{8}=256$ possible sign assignments; only the assignment in which all eight scores are negative produces $W^{+}=0$ .

Therefore, the null hypothesis was rejected for both TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) . These results provide evidence that the location of the dataset-level mean log MSE ratios is below zero. In other words, across the evaluated datasets, DPAL produces a geometric mean reconstructed price MSE ratio below one relative to the original adaptation objective. Both conclusions also remain significant under a Bonferroni correction for the two framework-specific tests.

##### A.3.2 Counts of Best and Second-Best Results

To further examine the consistency of each loss configuration across experimental settings, we counted the number of times each configuration achieved the best (lowest MSE) or second-best (second-lowest MSE) reconstructed price MSE in Tables A.1 . The counts were first obtained separately for each table and then aggregated across the eight datasets within each adaptation framework. Table A.3.2 summarizes the results.

Table: Table A.3.2 : Counts of best and second-best reconstructed price MSE results across the experimental settings under TAFAS and PETSA. Original denotes results using PETSA’s original adaptation objective; the TAFAS baseline is represented by the MSE-only configuration.

|  | TAFAS | PETSA |
| Configuration | Best | Second-best | Best | Second-best |
| Base | 1 | 3 | 2 | 3 |
| Original | – | – | 0 | 1 |
| mse | dir | price |  |  |  |  |
| ✓ |  |  | 0 | 0 | 0 | 0 |
|  | ✓ |  | 5 | 0 | 12 | 30 |
|  |  | ✓ | 5 | 0 | 4 | 1 |
| ✓ | ✓ |  | 14 | 10 | 0 | 19 |
| ✓ |  | ✓ | 1 | 3 | 0 | 0 |
|  | ✓ | ✓ | 56 | 46 | 23 | 62 |
| ✓ | ✓ | ✓ | 38 | 58 | 79 | 4 |

Under TAFAS ( Kim et al., 2025 ) , the combination of direction-aware and price calibration losses without the pointwise MSE term ranked among the top two, achieving 56 best and 46 second-best results. The complete DPAL objective followed closely, with 38 best and 58 second-best results. Under PETSA ( Medeiros et al., 2025 ) , complete DPAL achieved the largest number of best results (79), while the combination of direction-aware and price calibration losses appeared among the top two in 85 settings. By comparison, the MSE-only objective produced no best or second-best results under either framework, while the original PETSA ( Medeiros et al., 2025 ) objective produced only one second-best result.

Overall, the best and second-best results are concentrated among configurations that include both direction-aware and price calibration losses. The precise pattern differs between the two adaptation frameworks: the "direction plus price" configuration is most frequently the best under TAFAS ( Kim et al., 2025 ) , whereas the complete DPAL objective is most frequently the best under PETSA ( Medeiros et al., 2025 ) . These results indicate that the directional and price-based constraints make complementary contributions across the evaluated settings, while the most effective balance between these constraints and the pointwise MSE term may depend on the underlying adaptation framework.

#### A.4 Ablation on the Price Calibration Length

##### A.4.1 Experimental Configuration

To examine the effect of the truncation length $k$ in the price calibration loss, we conduct ablation experiments under both TAFAS ( Kim et al., 2025 ) and PETSA ( Medeiros et al., 2025 ) . The experiments use the same eight financial datasets as the main evaluation: Magnificent 7, S&P 500 Excerpt, Russell 2000 Excerpt, Technology Sector, Consumer Sector, Energy Sector, Nasdaq Mixed Excerpt, and Macro and Market Series. We consider five forecasting backbones, including iTransformer ( Liu et al., 2024 ) , PatchTST ( Nie et al., 2023 ) , DLinear ( Zeng et al., 2023 ) , FreTS ( Yi et al., 2023 ) , and OLS ( Toner and Darlow, 2024 ) , together with three forecast horizons $H\in\{60,90,120\}$ . Each adaptation framework consequently contains $8\times 5\times 3=120$ combinations of dataset, forecasting backbone, and horizon. All experiments retain the data splits, pretrained models, adaptation procedure, and matured-ground-truth protocol used in the main experiments. We also fix $\alpha=10^{5}$ , $\beta=10^{2}$ , and all other optimization settings so that $k$ is the only factor varied in this ablation.

We compare three settings of $k$ :

- • Reference setting: Follows the main experiments and uses $k=15$ for $H=60$ , and $k=30$ for both $H=90$ and $H=120$ .

- • Intermediate setting: Extends the calibration interval to $k=30$ , $45$ , and $60$ , respectively.

- • Full horizon setting: Uses $k=H$ , corresponding to $k=60$ , $90$ , and $120$ .

Across these settings, only the number of forecast steps included in the price calibration loss is changed. Performance is still evaluated using reconstructed price MSE over the complete $H$ -step forecast horizon.

##### A.4.2 Aggregation of Results

We first count how frequently each setting produces the lowest reconstructed price MSE. For every combination of dataset, forecasting backbone, and horizon, the three settings are compared using their unrounded MSE values, and one count is assigned to the setting with the lowest value. Because each adaptation framework contains 120 combinations, the lowest MSE counts of the three settings sum to 120.

Reconstructed price MSE can differ substantially in scale across datasets. Directly averaging the raw MSE values would consequently give greater influence to datasets with numerically larger errors. We instead use the reference setting as the baseline and aggregate the relative MSE ratios using a geometric mean.

Let $f\in\{\mathrm{TAFAS},\mathrm{PETSA}\}$ denote the adaptation framework, let $s$ denote a setting of $k$ , and let $E_{f,i}^{(s)}$ be the reconstructed price MSE of setting $s$ for experimental combination $i$ . The corresponding MSE under the reference setting is denoted by $E_{f,i}^{(\mathrm{ref})}$ . We first calculate the ratio:

$$r_{f,i}^{(s)}=\frac{E_{f,i}^{(s)}}{E_{f,i}^{(\mathrm{ref})}}.$$

The geometric mean relative change across all $N=120$ combinations is then defined as:

$$\Delta_{f}^{(s)}=\left[\left(\prod_{i=1}^{N}r_{f,i}^{(s)}\right)^{1/N}-1\right]\times 100\%=\left[\exp\left(\frac{1}{N}\sum_{i=1}^{N}\log\frac{E_{f,i}^{(s)}}{E_{f,i}^{(\mathrm{ref})}}\right)-1\right]\times 100\%.$$

This calculation first normalizes each alternative setting against the reference setting within the same experimental combination and then aggregates the resulting multiplicative ratios. It therefore avoids directly averaging raw MSE values from datasets with different numerical scales. The geometric mean is also a natural summary for relative ratios: $\Delta_{f}^{(s)}<0$ indicates a lower aggregate MSE than the reference setting, whereas $\Delta_{f}^{(s)}>0$ indicates a higher aggregate MSE. The relative change of the reference setting is zero by definition.

##### A.4.3 Results and Conclusions

As shown in Table A.4 , the reference setting achieves the largest number of lowest MSE results under both adaptation frameworks, with 51 cases under TAFAS ( Kim et al., 2025 ) and 56 under PETSA ( Medeiros et al., 2025 ) . The intermediate setting achieves 46 and 40 lowest MSE results, respectively, while increasing the geometric mean MSE relative to the reference by only 0.48% under TAFAS ( Kim et al., 2025 ) and 0.86% under PETSA ( Medeiros et al., 2025 ) . These small changes indicate that moderately extending $k$ has a limited overall effect and that the reference and intermediate settings form a relatively stable range of truncation lengths.

Table: Table A.4 : Ablation results for different price calibration lengths under TAFAS and PETSA. The lowest MSE count is the number of times that a setting achieves the lowest reconstructed price MSE among the 120 combinations of datasets, forecasting backbones, and horizons. Geometric mean changes are calculated relative to the reference setting.

|  | $k$ | Lowest MSE Count | Geometric Mean Change |
| Setting | $H=60$ | $H=90$ | $H=120$ | TAFAS | PETSA | TAFAS | PETSA |
| Reference | 15 | 30 | 30 | 51 | 56 | —– | —– |
| Intermediate | 30 | 45 | 60 | 46 | 40 | +0.48% | +0.86% |
| Full horizon | 60 | 90 | 120 | 23 | 24 | +4.95% | +7.82% |

When $k$ is extended to the complete forecast horizon, the number of lowest MSE results decreases to 23 under TAFAS ( Kim et al., 2025 ) and 24 under PETSA ( Medeiros et al., 2025 ) . At the same time, the geometric mean MSE increases by 4.95% and 7.82%, respectively. Compared with the intermediate setting, the full horizon setting produces a substantially larger increase under both frameworks. This common pattern indicates that applying the price calibration constraint to every forecast step is generally less reliable than restricting it to an early portion of the predicted trajectory. The observation is consistent with the cumulative nature of price reconstruction: prices at more distant steps incorporate longer sequences of compounded returns, so including the complete trajectory in the adaptation loss can make the calibration objective more sensitive to accumulated deviations.

Overall, the ablation results support using a truncated interval in the price calibration loss. The reference schedule $k=15/30/30$ produces the largest number of lowest MSE results under both adaptation frameworks and is more stable overall than calibration over the full horizon. Meanwhile, the small geometric mean differences between the reference and intermediate settings suggest that DPAL does not depend on a single sharply isolated value of $k$ but remains relatively stable across a short to intermediate calibration range. The reference setting can thus be regarded as a reasonable common choice across the evaluated datasets, forecasting backbones, and adaptation frameworks.

### Appendix B Additional Dataset Details

#### B.1 Data Source and Dataset Construction

All raw series were obtained from Yahoo Finance through the yfinance package. For each dataset, all constituent tickers were downloaded jointly using yf.download() with auto_adjust=True . Under this setting, the values returned in the Close field are adjusted for corporate actions and therefore differ from unadjusted closing prices. No separate Adj Close field was used.

The requested download interval was specified as $[2012\text{-}05\text{-}18,2026\text{-}01\text{-}01)$ , with the starting date included and the ending date excluded. The latest retained observation is therefore December 31, 2025. After return construction and missing value removal, seven datasets begin on May 21, 2012, whereas the Russell 2000 Excerpt begins on August 20, 2013 because complete observations for all of its constituent series were not available before that date.

The equity panels were manually curated. The panels were designed to cover different market segments and company types while retaining sufficiently long historical series and avoiding repeated tickers across the seven equity datasets. Table B.1 reports their complete composition.

Table: Table B.1 : The eight datasets.

| Dataset | Number of series | Tickers |
| Magnificent 7 | 7 | AAPL, MSFT, GOOG, AMZN, NVDA, META, TSLA |
| S&P 500 Excerpt | 20 | CDNS, SNPS, KLAC, LRCX, MCHP, MPWR, MRVL, ON, TER, SWKS, STX, WDC, FSLR, VRSN, MSI, ROP, FICO, PTC, ZBRA, APH |
| Russell 2000 Excerpt | 20 | MHO, WWD, FIX, AAON, ENSG, CASY, KBR, TEX, UGI, WBS, AWI, SPSC, CNO, MTG, PII, SAIA, LECO, THG, MUSA, FIBK |
| Technology Sector | 20 | AMD, AVGO, QCOM, INTC, ASML, MU, TXN, AMAT, CRM, ADBE, TSM, ORCL, ADSK, FTNT, AKAM, NTAP, INTU, NFLX, CSCO, IBM |
| Consumer Sector | 20 | CHD, KO, PEP, WMT, COST, TGT, CL, KMB, GIS, HSY, MKC, EL, KR, HD, LOW, TJX, NKE, MCD, SBUX, YUM |
| Energy Sector | 20 | XOM, CVX, COP, OXY, EOG, DVN, APA, EQT, RRC, MUR, SM, NOG, SLB, HAL, NOV, VLO, MPC, PSX, KMI, WMB |
| Nasdaq Mixed Excerpt | 20 | AMGN, GILD, ISRG, IDXX, CMCSA, TTWO, CSX, PCAR, CTAS, FAST, CPRT, ODFL, ADP, VRSK, CSGP, NXPI, CTSH, BKNG, MELI, AEP |
| Macro and Market Series | 10 | ^GSPC, ^NDX, ^MID, ^RUT, ^DJI, ^TNX, DX-Y.NYB, GC=F, CL=F, ^NYA |

#### B.2 Return Construction and Trading Date Alignment

Let $P_{t,c}$ denote the adjusted closing value of series $c$ on trading day $t$ . The daily simple return was calculated using pct_change() :

$$r_{t,c}=\frac{P_{t,c}-P_{t-1,c}}{P_{t-1,c}}=\frac{P_{t,c}}{P_{t-1,c}}-1.$$

Returns were stored as decimal values rather than percentages.

Within each dataset, the series were first aligned on the combined date index produced by the joint download. Under the default behavior of pct_change() used when generating the datasets, a missing price observation was replaced by the most recently available price before the percentage change was calculated. Consequently, if a series had no new quotation on a date retained in the combined index, its return on that date was effectively recorded as zero. Rows that still contained undefined returns after this operation, including the initial row required by the percentage change calculation and rows preceding the availability of every constituent series, were removed using dropna() .

This procedure ensures that all series within a dataset share the same final date index, but it does not require every retained date to be a trading day for every constituent. This distinction is most relevant to the Macro and Market Series dataset because its equity indices, Treasury yield index, currency index, and commodity futures series follow different trading calendars. A date could therefore be retained when only some markets reported new values; series without a new observation on that date receive a zero return under the procedure described above. Thus, the Macro and Market Series dataset is not restricted to the strict intersection of the trading dates shared by all ten series.

Each constituent was retained as a separate target series. No aggregation across series or constituent weighting was applied.

#### B.3 Chronological Data Splits

Each dataset was arranged chronologically and divided into training, validation, and test portions using an approximate ratio of $7{:}1{:}2$ . For a dataset containing $N$ return observations, the numbers assigned to the training and test portions were calculated as

$$N_{\mathrm{train}}=\left\lfloor 0.7N\right\rfloor,\qquad N_{\mathrm{test}}=\left\lfloor 0.2N\right\rfloor,$$ (16)

and the remaining observations were assigned to validation:

$$N_{\mathrm{val}}=N-N_{\mathrm{train}}-N_{\mathrm{test}}.$$ (17)

The first $N_{\mathrm{train}}$ observations formed the training portion, the following $N_{\mathrm{val}}$ observations formed the validation portion, and the final $N_{\mathrm{test}}$ observations formed the test portion. Table B.2 reports the resulting date ranges, with both endpoints included.

Table: Table B.2: Chronological data splits. The number in parentheses is the number of return observations in the corresponding portion.

| Dataset | Training | Validation | Test |
| Magnificent 7 | 2012-05-21–2021-11-24 (2,396) | 2021-11-26–2023-04-11 (344) | 2023-04-12–2025-12-31 (684) |
| S&P 500 Excerpt | 2012-05-21–2021-11-24 (2,396) | 2021-11-26–2023-04-11 (344) | 2023-04-12–2025-12-31 (684) |
| Russell 2000 Excerpt | 2013-08-20–2022-04-11 (2,177) | 2022-04-12–2023-07-11 (312) | 2023-07-12–2025-12-31 (622) |
| Technology Sector | 2012-05-21–2021-11-24 (2,396) | 2021-11-26–2023-04-11 (344) | 2023-04-12–2025-12-31 (684) |
| Consumer Sector | 2012-05-21–2021-11-24 (2,396) | 2021-11-26–2023-04-11 (344) | 2023-04-12–2025-12-31 (684) |
| Energy Sector | 2012-05-21–2021-11-24 (2,396) | 2021-11-26–2023-04-11 (344) | 2023-04-12–2025-12-31 (684) |
| Nasdaq Mixed Excerpt | 2012-05-21–2021-11-24 (2,396) | 2021-11-26–2023-04-11 (344) | 2023-04-12–2025-12-31 (684) |
| Macro and Market Series | 2012-05-21–2021-11-29 (2,399) | 2021-11-30–2023-04-13 (344) | 2023-04-14–2025-12-31 (685) |

When constructing validation and test samples, the data loader additionally included the 60 observations immediately preceding each split boundary as historical input. These observations provided the lookback context required by the forecasting models, while the corresponding prediction targets began at the validation or test boundary. The additional context therefore did not alter the target date ranges reported in Table B.2 . The standardization statistics were fitted using only the training portion and were subsequently applied to the validation and test portions.
