# شکل خطای فاز در تخمین کانال Widely-Linear برای Cell-Free Massive MIMO

این مخزن، کد و داده‌های بازتولید نتایج مقاله‌ی زیر را در اختیار داوران و پژوهشگران قرار می‌دهد:

> **Phase-Error Shape in Widely-Linear Channel Estimation for Cell-Free Massive MIMO**  
> Pooya Hejazi and Masoud Sabaei  
> Amirkabir University of Technology

نسخه‌ی مقاله و فایل PDF آن در این مخزن قرار ندارد و از مسیر فرآیند داوری ژورنال در دسترس است. سیاست این مخزن این است که فقط **کد، پیکربندی و داده‌های عددی لازم برای بازتولید** در آن نگهداری شود.

---

## درباره‌ی مقاله

در سیستم‌های Cell-Free Massive MIMO، خطای فاز باقی‌مانده پس از ردیابی معمولاً فقط با اولین گشتاور دایره‌ای، یعنی

$$
m_1 = \mathbb{E}[\cos\varepsilon],
$$

خلاصه می‌شود. مقاله نشان می‌دهد که این خلاصه برای تخمین کانال Widely-Linear (WL) کافی نیست؛ زیرا تخمین‌گر WL به دومین گشتاور دایره‌ای و شبه‌کوواریانس

$$
c = m_2 - m_1^2,
\qquad
m_2 = \mathbb{E}[\cos 2\varepsilon],
$$

نیز وابسته است.

در $m_1$ ثابت، مقدار $c$ یک درجه‌ی آزادی مستقل روی بازه‌ی زیر است:

$$
-(1-m_1^2) \le c \le 1-m_1^2.
$$

تخمین‌گر complex-affine که در مقاله **B2** نامیده می‌شود فقط از $m_1$ استفاده می‌کند، در حالی که تخمین‌گر Widely-Linear یا **B1** از هر دو $m_1$ و $c$ استفاده می‌کند. بنابراین «شکل» توزیع خطای فاز، و نه فقط واریانس آن، روی بهره‌ی WL اثر می‌گذارد.

---

## نتایج کلیدی

- برای $0<m_1<1$ ثابت شده است که سه قانون فاز **Binary، Uniform و Wrapped Gaussian** از نظر $|c|$ و در نتیجه بهره‌ی WL اسکالر، ترتیب اکید دارند.
- قانون **von Mises** به‌صورت عددی روی یک شبکه‌ی ۲۰٬۰۰۱ نقطه‌ای بررسی شده و ادعای قضیه برای آن مطرح نمی‌شود.
- یک پنجره‌ی کالیبراسیون به‌ازای هر زنجیره‌ی RF پیشنهاد شده است. روی قوانین فاز اعلام‌شده:
  - در $M=1024$، نسبت بهره‌ی حفظ‌شده برای B/U/G برابر است با حدود **0.984 / 0.966 / 0.961**.
  - نقطه‌ی عبور ۹۹٪ برای B/U/G برابر است با حدود **2,216 / 4,141 / 3,898** فریم.
  - این اعداد تخمین‌های Monte Carlo هستند و تضمین عملیاتی محسوب نمی‌شوند.
- در سه اجرای مستقل Cell-Free با pilot reuse و RZF، اندازه‌ی اثر شکل فاز برابر است با **2.1196، 2.2079 و 2.0814** و ترتیب کامل در هر ۶۴ topology برقرار است.
- تحلیل pooled برای حالت collapse مقدار **1.0590** با بازه‌ی bootstrap برابر **[1.0348, 1.0861]** به دست می‌دهد.
- مطالعه‌ی downlink فقط برای **Maximum-Ratio precoding** انجام شده و ترتیب کامل downlink ادعا نمی‌شود.

---

## ساختار مخزن

```text
analysis/
  phase_laws.py                  # قوانین فاز، لنز feasible و بررسی vM
  calibration.py                 # محاسبات پایه‌ی calibration و MSE
  calibration_law.py             # Monte Carlo روی قوانین واقعی فاز
  requirements.txt
  results/
    cp1_phase_laws.json          # خروجی بررسی قوانین و ترتیب
    cp2_calibration_law.json     # خروجی کالیبراسیون، capture و M*

reproducibility/
  system_run/                    # اجرای اصلی frozen و داده‌های مرجع
  expanded_run_v02/              # اجرای گسترده‌شده: kappa و sensitivity
  v08_run/                       # بسته‌ی v08، replay و vendor code
  standard_channel/              # بررسی کانال استاندارد 3GPP UMa
  downlink/                      # بررسی downlink با MR precoding
  pooled_collapse.py             # تحلیل pooled سه اجرای مستقل
  pooled_collapse.json
```

فایل‌های `ENVIRONMENT_LOCK.json`، `requirements.lock`، پیکربندی‌ها، checksumها و manifestها برای حفظ provenance هر بسته در مخزن نگه داشته شده‌اند.

---

## نصب محیط

Python 3.12 توصیه می‌شود.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r analysis\requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
```

بسته‌های frozen از نسخه‌های مشخص‌شده در `requirements.lock` استفاده می‌کنند. نسخه‌های مرجع عبارت‌اند از:

```text
Python 3.12
numpy  2.5.2
scipy  1.18.1
PyYAML 6.0.3
```

---

## نحوه‌ی اجرا و دریافت خروجی‌ها

همه‌ی دستورها باید **از ریشه‌ی مخزن** اجرا شوند؛ یعنی پوشه‌ای که `README.md` و پوشه‌های `analysis/` و `reproducibility/` در آن قرار دارند.

پس از فعال‌سازی محیط مجازی، در Windows استفاده از مسیر صریح مفسر نیز مجاز است:

```powershell
.venv\Scripts\python.exe -B <script.py>
```

در Linux یا macOS:

```bash
python -B <script.py>
```

### گام ۱: اجرای محاسبات اسکالر

```bash
python -B analysis/phase_laws.py
python -B analysis/calibration_law.py
```

این دو دستور خروجی‌های زیر را می‌سازند یا بازنویسی می‌کنند:

| دستور | خروجی فایل | خروجی ترمینال |
|---|---|---|
| `analysis/phase_laws.py` | `analysis/results/cp1_phase_laws.json` | نتیجه‌ی ترتیب قوانین، بررسی sinc و مقایسه‌ی vM |
| `analysis/calibration_law.py` | `analysis/results/cp2_calibration_law.json` | capture، خطا، bias، projection و $M^\star$ برای هر قانون |

`calibration_law.py` از seedهای پایدار مشتق‌شده از SHA-256 استفاده می‌کند. اجرای دوباره‌ی آن باید JSON یکسان و byte-identical تولید کند. این اسکریپت چند ده ثانیه زمان می‌برد و سنگین‌ترین بخش آن Monte Carlo با ۲۰۰۰ replicate برای هر قانون و هر طول پنجره است.

### گام ۲: تحلیل خروجی‌های frozen

```bash
python -B reproducibility/expanded_run_v02/analyze_expanded.py
python -B reproducibility/system_run/analyze_replication.py
python -B reproducibility/system_run/robustness_replication.py
```

| دستور | ورودی | خروجی |
|---|---|---|
| `analyze_expanded.py` | `expanded_run_v02/output/E10_shape_output.json` | فایل `expanded_run_v02/output/expanded_analysis.json` و گزارش اثرها در ترمینال |
| `system_run/analyze_replication.py` | `system_run/output/E10_shape_output.json` | فقط گزارش ordering، collapse، controls و اثرها در ترمینال |
| `system_run/robustness_replication.py` | `system_run/output/E10_shape_output.json` | فقط SE، توان، MDD، jackknife و تحلیل حساسیت در ترمینال |

دو اسکریپت `system_run` فایلی نمی‌سازند. اگر می‌خواهید خروجی ترمینال آن‌ها را ذخیره کنید:

```powershell
python -B reproducibility/system_run/analyze_replication.py | Tee-Object -FilePath replication_report.txt
python -B reproducibility/system_run/robustness_replication.py | Tee-Object -FilePath robustness_report.txt
```

### گام ۳: بازتولید downlink و کانال استاندارد

```bash
python -B reproducibility/downlink/run_downlink.py
python -B reproducibility/standard_channel/run_standard.py
```

| دستور | خروجی فایل | نکته |
|---|---|---|
| `downlink/run_downlink.py` | `reproducibility/downlink/results.json` | شامل deterministic self-test، میانگین SE، بازه‌های paired و تعداد topologyهای مثبت |
| `standard_channel/run_standard.py` | `reproducibility/standard_channel/results.json` | شامل دو variant کانال UMa و بررسی‌های validation |

اجرای downlink معمولاً حدود یک دقیقه و اجرای standard-channel چند دقیقه زمان می‌برد. این زمان‌ها به سخت‌افزار وابسته‌اند.

### گام ۴: تحلیل pooled

```bash
python -B reproducibility/pooled_collapse.py
```

این دستور سه فایل زیر را می‌خواند:

```text
reproducibility/v08_run/run/output/E10_shape_output.json
reproducibility/system_run/output/E10_shape_output.json
reproducibility/expanded_run_v02/output/E10_shape_output.json
```

و نتیجه را در `reproducibility/pooled_collapse.json` می‌نویسد. مقادیر مرجع عبارت‌اند از:

```text
M090: pooled_effect = 2.1372452505
      interval      = [2.0789193293, 2.2074038563]
M099: pooled_effect = 1.0589764846
      interval      = [1.0347764372, 1.0861185945]
```

### گام ۵: مشاهده‌ی یک خروجی JSON

برای چاپ مرتب یک فایل JSON:

```bash
python -B -c "import json; p='analysis/results/cp2_calibration_law.json'; print(json.dumps(json.load(open(p, encoding='utf-8')), indent=2, ensure_ascii=False))"
```

برای دیدن مستقیم بخش خلاصه‌ی کالیبراسیون:

```bash
python -B -c "import json; d=json.load(open('analysis/results/cp2_calibration_law.json', encoding='utf-8')); print(json.dumps(d['summary'], indent=2))"
```

اگر `jq` نصب باشد، نمونه‌های کوتاه‌تر نیز قابل استفاده‌اند:

```bash
jq '.summary' analysis/results/cp2_calibration_law.json
jq '.M099.pooled_effect, .M099.pooled_interval' reproducibility/pooled_collapse.json
```

### چه فایل‌هایی بازنویسی می‌شوند؟

اجرای دستورهای بالا فقط خروجی‌های تحلیلی زیر را بازنویسی می‌کند:

- `analysis/results/cp1_phase_laws.json`
- `analysis/results/cp2_calibration_law.json`
- `reproducibility/expanded_run_v02/output/expanded_analysis.json`
- `reproducibility/downlink/results.json`
- `reproducibility/standard_channel/results.json`
- `reproducibility/pooled_collapse.json`

فایل‌های `E10_shape_output.json` خروجی‌های علمی frozen هستند و دستورهای بالا آن‌ها را بازنویسی نمی‌کنند. اسکریپت `run/E10_shape_run.py` را اجرا نکنید؛ آن اسکریپت driver اصلی اجرای frozen است، نه تحلیل خروجی.

---

## بازاجرای کامل v08

بازاجرای کامل بسته‌ی v08 با دستور زیر انجام می‌شود:

```bash
python -B reproducibility/v08_run/replay.py
```

این replay تقریباً ۷۰ دقیقه زمان می‌برد. مسیرهای تازه را فقط در محدوده‌ی کنترل‌شده می‌سازد، در صورت وجود `src/` از شروع امتناع می‌کند و فقط مسیرهایی را حذف می‌کند که خودش ایجاد کرده است.

خروجی‌های replay:

| مسیر | محتوا |
|---|---|
| `reproducibility/v08_run/replay_output/E10_shape_output.json` | خروجی تازه‌ی replay برای مقایسه |
| `reproducibility/v08_run/REPLAY_RESULT_ROUND3.json` | گزارش مقایسه‌ی ۲۰ بخش علمی، تعداد recordها و mismatchها |

دستور زیر را برای بسته‌های frozen اجرا نکنید:

```text
reproducibility/v08_run/run/E10_shape_run.py
reproducibility/system_run/E10_shape_run.py
reproducibility/expanded_run_v02/run/E10_shape_run.py
```

این فایل‌ها driverهای اصلی آزمایش هستند و اجرای مجدد آن‌ها بخشی از رویه‌ی تحلیل معمول نیست.

---

## داده‌های مرجع

| فایل | محتوا |
|---|---|
| `analysis/results/cp1_phase_laws.json` | بررسی تحلیلی و عددی ترتیب قوانین فاز |
| `analysis/results/cp2_calibration_law.json` | capture، covariance، bias، projection و $M^\star$ |
| `reproducibility/system_run/output/E10_shape_output.json` | اجرای اصلی frozen |
| `reproducibility/expanded_run_v02/output/E10_shape_output.json` | اجرای expanded و sensitivity |
| `reproducibility/pooled_collapse.json` | تحلیل pooled سه draw |
| `reproducibility/downlink/results.json` | نتایج downlink و deterministic self-test |
| `reproducibility/standard_channel/results.json` | نتایج کانال استاندارد UMa |

---

## دامنه و محدودیت‌ها

برای جلوگیری از تفسیر بیش از حد نتایج، نکات زیر بخشی از دامنه‌ی ادعاهای مقاله هستند:

- ترتیب تحلیلی B/U/G برای $0<m_1<1$ اثبات شده است؛ ترتیب von Mises فقط عددی و روی شبکه‌ی محدود بررسی شده است.
- نتایج Cell-Free سنتزی هستند و ادعای اندازه‌گیری سخت‌افزاری یا multi-site وجود ندارد.
- بسته‌های expanded، خطای فرضی $c$ را تزریق می‌کنند و یک بررسی sensitivity هستند، نه اعتبارسنجی یک tracker حلقه‌بسته.
- کالیبراسیون اسکالر یک کران و تحلیل Monte Carlo است و پنجره‌ی به‌دست‌آمده تضمین عملیاتی عمومی نیست.
- قانون Binary یک حالت مرزی است و بخش قابل‌توجهی از پنجره‌های آن روی مرز feasible projection می‌شود.
- بررسی downlink فقط برای MR precoding است؛ بازه‌های paired ترتیب کامل را resolve نمی‌کنند.
- کانال standard-channel به‌صورت frequency-flat مدل شده و delay dispersion ندارد.

---

## استناد

اگر از این کد یا داده‌ها در پژوهش خود استفاده می‌کنید، لطفاً به مقاله‌ی زیر ارجاع دهید:

```bibtex
@article{hejazi2026phase,
  author  = {Pooya Hejazi and Masoud Sabaei},
  title   = {Phase-Error Shape in Widely-Linear Channel Estimation for Cell-Free Massive MIMO},
  journal = {IEEE Transactions on Wireless Communications},
  year    = {2026},
  note    = {Manuscript}
}
```

اطلاعات نهایی جلد، شماره و DOI پس از انتشار به‌روزرسانی خواهد شد.
