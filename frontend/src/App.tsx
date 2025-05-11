import { useState } from 'react';
import Plot from 'react-plotly.js';

/* ---------- типы ответов бэкенда ---------- */
interface TrainResp {
  metrics: { MAE: number; MAPE: number };
  hist:   { Year: number[]; Value: number[] };
  test:   { Year: number[]; y_true: number[]; y_pred: number[] };
}

interface ForecastResp {
  Year: number[];
  y_pred: number[];
}

/* ---------- хелпер отправки FormData ---------- */
async function postForm(url: string, fd: FormData) {
  const r = await fetch(url, { method: 'POST', body: fd });
  return r.json();
}

/* ---------- сопоставляем произвольные файлы нужным полям ---------- */
function buildFormData(
  files: FileList,
  model: string,
  extra: Record<string, string> = {}
) {
  const fd = new FormData();
  const map: Record<string, string> = {
    birth: 'births',
    death: 'deaths',
    pop:   'population',
    mig:   'migration',
  };

  Array.from(files).forEach(f => {
    const lname = f.name.toLowerCase();
    const key = Object.keys(map).find(k => lname.includes(k));
    if (key) fd.append(map[key], f);
  });

  // проверка всех 4
  if (!['births', 'deaths', 'population', 'migration'].every(k => fd.has(k))) {
    throw new Error('Не удалось опознать все 4 CSV (Birth/Death/Population/Migration)');
  }

  fd.append('model', model);
  Object.entries(extra).forEach(([k, v]) => fd.append(k, v));
  return fd;
}

/* ================================================================== */
export default function App() {
  const [model, setModel] = useState<'sarimax' | 'sarimax_pop'>('sarimax');
  const [files, setFiles] = useState<FileList | null>(null);

  const [trainData, setTrainData] = useState<TrainResp | null>(null);
  const [forecast, setForecast]   = useState<ForecastResp | null>(null);

  const [years, setYears] = useState(5);

  /* ----------------------- обработчики ---------------------------- */
  const handleTrain = async () => {
    if (!files) return alert('Выберите 4 CSV-файла');
    try {
      const fd = buildFormData(files, model);
      const resp = (await postForm('/upload-train/', fd)) as TrainResp;
      setTrainData(resp);
      setForecast(null);
    } catch (e: any) {
      alert(e.message);
    }
  };

  const handleForecast = async () => {
    if (!files) return alert('Выберите 4 CSV-файла');
    try {
      const fd = buildFormData(files, model, { years: String(years) });
      const resp = (await postForm('/forecast/', fd)) as ForecastResp;
      setForecast(resp);
    } catch (e: any) {
      alert(e.message);
    }
  };

  /* ----------------------- UI ------------------------------------- */
  return (
    <div style={{ padding: 24, maxWidth: 900 }}>
      <h1>Demography Forecast MVP</h1>

      {/* загрузка файлов + выбор модели */}
      <input
        type="file"
        accept=".csv"
        multiple
        onChange={e => setFiles(e.target.files)}
        style={{ marginRight: 12 }}
      />

      <select value={model} onChange={e => setModel(e.target.value as any)}>
        <option value="sarimax">SARIMAX (Births)</option>
        <option value="sarimax_pop">SARIMAX-POP (Population)</option>
      </select>

      <button style={{ marginLeft: 12 }} onClick={handleTrain}>
        Train
      </button>

      {/* прогноз */}
      <span style={{ marginLeft: 24 }}>Forecast:</span>
      <input
        type="range"
        min={1}
        max={22}
        value={years}
        onChange={e => setYears(+e.target.value)}
        style={{ width: 120, verticalAlign: 'middle', margin: '0 8px' }}
      />
      <span>{years}</span>
      <button style={{ marginLeft: 8 }} onClick={handleForecast}>
        Predict
      </button>

      {/* ---------------------------------------------------------------- */}
      {trainData && (
        <>
          {/* таблица метрик */}
          <table style={{ borderCollapse: 'collapse', marginTop: 20 }}>
            {Object.entries(trainData.metrics).map(([k, v]) => (
              <tr key={k}>
                <td style={{ border: '1px solid #aaa', padding: 4 }}>{k}</td>
                <td style={{ border: '1px solid #aaa', padding: 4 }}>
                  {v.toFixed(2)}
                </td>
              </tr>
            ))}
          </table>

          {/* график (Births или Population в зависимости от модели) */}
          <h3 style={{ marginTop: 28 }}>
            {model === 'sarimax' ? 'Births' : 'Population'}
          </h3>
          <Plot
            data={[
              {
                x: trainData.hist.Year,
                y: trainData.hist.Value,
                mode: 'lines',
                name: 'История',
                line: { color: 'black' },
              },
              {
                x: trainData.test.Year,
                y: trainData.test.y_pred,
                mode: 'lines',
                name: 'Test forecast',
                line: { dash: 'dash', color: 'orange' },
              },
              forecast && {
                x: forecast.Year,
                y: forecast.y_pred,
                mode: 'lines',
                name: 'Future forecast',
                line: { dash: 'dot' },
              },
            ].filter(Boolean)}
            layout={{
              xaxis: { title: 'Year' },
              yaxis: { title: model === 'sarimax' ? 'Births' : 'Population' },
              shapes: [
                {
                  type: 'line',
                  x0: 2021.5,
                  x1: 2021.5,
                  y0: 0,
                  y1: 1,
                  yref: 'paper',
                  line: { dash: 'dot', color: 'grey' },
                },
              ],
              legend: { orientation: 'h', x: 0.25, y: 1.15 },
              margin: { t: 50, r: 40, l: 60, b: 50 },
            }}
            style={{ width: '100%', height: 450 }}
            config={{ responsive: true }}
          />
        </>
      )}
    </div>
  );
}
