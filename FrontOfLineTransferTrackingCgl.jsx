import { translate } from "shared/base/translation";
import Translation from "translation.js";

const t = translate(Translation);

const FrontOfLineTransferTrackingCgl = () => {
  // Basit grid data
  const gridData = Array.from({ length: 6 }, (_, row) =>
    Array.from({ length: 25 }, (_, col) => ({
      id: `${row}-${col}`,
      number: Math.floor(Math.random() * 999).toString().padStart(3, '0'),
      active: Math.random() > 0.6
    }))
  );

  return (
    <div className="w-full h-screen bg-gray-300">
      {/* Header */}
      <div className="bg-blue-600 text-white p-3 text-center">
        <h1 className="text-xl font-bold">IA1 THE ENTRY AREA LOGIST</h1>
        <div className="text-sm mt-1">2025/1/18 11:22:37</div>
      </div>

      <div className="flex p-4 gap-4 h-full">
        {/* Sol Panel */}
        <div className="w-48 space-y-3">
          <div className="bg-gray-100 border p-3">
            <button className="w-full bg-green-500 text-white p-2 mb-2 rounded">START</button>
            <button className="w-full bg-red-500 text-white p-2 mb-2 rounded">PAUSE</button>
            <div className="bg-black text-green-400 p-2 text-center font-mono">1001</div>
          </div>
          
          <div className="bg-gray-100 border p-3">
            <div className="bg-green-500 text-white p-2 text-center mb-2">AUTOMATIC</div>
            <div className="flex gap-1 mb-2">
              <div className="flex-1 bg-green-500 text-white p-1 text-center text-sm">+0</div>
              <div className="flex-1 bg-green-500 text-white p-1 text-center text-sm">+0</div>
            </div>
            <div className="bg-pink-200 p-2 text-center">000</div>
          </div>
        </div>

        {/* Ana Grid */}
        <div className="flex-1 bg-white border p-4">
          <div className="grid gap-1 mb-4" style={{ gridTemplateColumns: 'repeat(25, 1fr)' }}>
            {gridData.map((row, rowIndex) =>
              row.map((item, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`
                    border h-8 flex items-center justify-center text-xs font-mono
                    ${item.active ? 'bg-blue-400 border-blue-600' : 'bg-gray-200 border-gray-400'}
                  `}
                >
                  {item.active && item.number}
                </div>
              ))
            )}
          </div>
          
          {/* Alt butonlar */}
          <div className="flex justify-center gap-3 mt-4">
            <button className="px-4 py-2 bg-gray-400 text-white rounded">RETURN</button>
            <button className="px-4 py-2 bg-blue-500 text-white rounded">CGL1</button>
            <button className="px-4 py-2 bg-blue-500 text-white rounded">CGL2</button>
            <button className="px-4 py-2 bg-blue-500 text-white rounded">CGL3</button>
            <button className="px-4 py-2 bg-yellow-500 text-white rounded">EXIT</button>
          </div>
        </div>

        {/* Sağ Panel */}
        <div className="w-48 space-y-3">
          <div className="bg-gray-100 border p-3">
            <div className="text-sm mb-2">CGL1_PX</div>
            <div className="w-full h-3 bg-yellow-400 border mb-2"></div>
            <div className="text-sm mb-2">CGL2_PX</div>
            <div className="w-full h-3 bg-yellow-400 border"></div>
          </div>
          
          <div className="bg-gray-100 border p-3">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-6 h-6 bg-red-500 rounded-full"></div>
              <span className="text-xs">N011_CAR</span>
            </div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-6 h-6 bg-red-500 rounded-full"></div>
              <span className="text-xs">N012_CAR</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 bg-blue-500 rounded-full"></div>
              <span className="text-xs">N013_CAR</span>
            </div>
          </div>
          
          <div className="bg-gray-100 border p-3">
            <button className="w-full bg-green-500 text-white p-2 mb-2 rounded">START</button>
            <button className="w-full bg-red-500 text-white p-2 mb-2 rounded">PAUSE</button>
            <div className="bg-green-500 text-white p-2 text-center">1216</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FrontOfLineTransferTrackingCgl;