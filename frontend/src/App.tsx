import { useState } from 'react';
import ModelForm from './components/ModelForm';
import MetricsTable from './components/MetricsTable';
import ForecastChart from './components/ForecastChart';

function DoubleChart({
  model,
  future,
}: {
  model: string;
  future: { Year: number[]; y_pred: number[] } | null;
}) {
  // будущий прогноз относится либо к Births-модели, либо к Population
  const futureBirth =
    model.endsWith('_pop') ? null : future;
  const futurePop =
    model.endsWith('_pop') ? future : null;

  return (
    <>
      <h3>Births</h3>
      <ForecastChart
        model={model === 'sarimax_pop' ? 'sarimax' : model}
        yTitle="Births"
        future={futureBirth}
      />

      <h3 style={{ marginTop: 32 }}>Population</h3>
      <ForecastChart
        model="sarimax_pop"
        yTitle="Population"
        future={futurePop}
      />
    </>
  );
}

export default function App() {
  const [selected, setSelected] = useState('sarimax');
  const [trained, setTrained] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const [future, setFuture] =
    useState<{ Year: number[]; y_pred: number[] } | null>(null);

  const handleTrained = () => {
    setTrained(selected);
    setFuture(null);          // сбросить предыдущий forecast
    setVersion(v => v + 1);   // форс-рендер
  };

  return (
    <div style={{ padding: 20 }}>
      <h1 style={{ marginBottom: 12 }}>Demography Forecast MVP</h1>

      <ModelForm
        selected={selected}
        onSelect={setSelected}
        onTrained={handleTrained}
        onForecast={setFuture}
      />

      {trained && (
        <>
          <MetricsTable
            key={`${trained}-tbl-${version}`}
            model={trained}
          />
          <DoubleChart
            key={`${trained}-plot-${version}`}
            model={trained}
            future={future}
          />
        </>
      )}
    </div>
  );
}
