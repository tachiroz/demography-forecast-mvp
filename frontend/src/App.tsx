import { useState } from 'react';
import ModelForm from './components/ModelForm';
import MetricsTable from './components/MetricsTable';
import ForecastChart from './components/ForecastChart';

export default function App() {
  const [selected, setSelected]   = useState('sarimax');   // что в селекте
  const [trained,  setTrained]    = useState<string|null>(null); // что уже обучили
  const [version,  setVersion]    = useState(0);           // заставляет перерисовать

  const handleTrained = () => {
    setTrained(selected);          // теперь эта модель стала «активной»
    setVersion(v => v + 1);        // меняем key, чтобы форс-перерисовать
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Demography Forecast MVP</h1>

      <ModelForm
        selected={selected}
        onSelect={setSelected}
        onTrained={handleTrained}
      />

      {trained && (
        <>
          <MetricsTable key={trained + version} model={trained} />
          <ForecastChart key={trained + version} model={trained} />
        </>
      )}
    </div>
  );
}
