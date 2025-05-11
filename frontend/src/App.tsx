import { useState } from 'react';
import ModelForm from './components/ModelForm';
import MetricsTable from './components/MetricsTable';
import ForecastChart from './components/ForecastChart';

function DoubleChart({ model }: { model: string }) {
  return (
    <>
      {/* Births график */}
      <h3>Births</h3>
      <ForecastChart
        model={model === 'sarimax_pop' ? 'sarimax' : model}
        yTitle="Births"
      />

      {/* Population график */}
      <h3 style={{ marginTop: 32 }}>Population</h3>
      <ForecastChart
        model="sarimax_pop"          // всегда population-модель
        yTitle="Population"
      />
    </>
  );
}

export default function App() {
  const [selected, setSelected] = useState('sarimax');
  const [trained, setTrained]   = useState<string | null>(null);
  const [version, setVersion]   = useState(0);         // принудительный перерендер

  const handleTrained = () => {
    setTrained(selected);        // какая модель обучилась
    setVersion(v => v + 1);      // заставим дочерние ключи обновиться
  };

  return (
    <div style={{ padding: 20 }}>
      <h1 style={{ marginBottom: 12 }}>Demography Forecast MVP</h1>

      <ModelForm
        selected={selected}
        onSelect={setSelected}
        onTrained={handleTrained}
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
          />
        </>
      )}
    </div>
  );
}
