import { translate } from "shared/base/translation";
import Translation from "translation.js";

const t = translate(Translation);

const FrontOfLineTransferTrackingCgl = () => {
  // Grid data for the main conveyor system
  const gridData = Array.from({ length: 8 }, (_, row) =>
    Array.from({ length: 30 }, (_, col) => ({
      id: `${row}-${col}`,
      status: Math.random() > 0.7 ? 'occupied' : Math.random() > 0.5 ? 'moving' : 'empty',
      number: Math.floor(Math.random() * 9999).toString().padStart(4, '0')
    }))
  );

  const StatusButton = ({ status, children, className = "" }) => (
    <button className={`px-3 py-2 rounded font-semibold text-sm transition-all ${className} ${
      status === 'start' ? 'bg-green-500 text-white hover:bg-green-600' :
      status === 'pause' ? 'bg-red-500 text-white hover:bg-red-600' :
      status === 'active' ? 'bg-blue-500 text-white' :
      'bg-gray-300 text-gray-700 hover:bg-gray-400'
    }`}>
      {children}
    </button>
  );

  const GridCell = ({ item, index }) => {
    const getColor = () => {
      switch (item.status) {
        case 'occupied': return 'bg-blue-400 border-blue-600';
        case 'moving': return 'bg-green-400 border-green-600';
        case 'empty': return 'bg-gray-200 border-gray-400';
        default: return 'bg-gray-200 border-gray-400';
      }
    };

    return (
      <div className={`
        relative border-2 ${getColor()} 
        min-h-[40px] flex items-center justify-center 
        text-xs font-mono transition-all hover:scale-105
      `}>
        {item.status !== 'empty' && (
          <span className="text-black font-bold">{item.number}</span>
        )}
        {item.status === 'moving' && (
          <div className="absolute inset-0 flex items-center justify-center">
            <svg width="16" height="16" className="animate-spin">
              <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="2" fill="none" />
              <path d="M14 8a6 6 0 0 0-6-6" stroke="currentColor" strokeWidth="2" />
            </svg>
          </div>
        )}
      </div>
    );
  };

  const ControlPanel = ({ title, children, className = "" }) => (
    <div className={`bg-gray-100 border-2 border-gray-400 rounded p-3 ${className}`}>
      <h3 className="font-bold text-sm mb-2 text-center">{title}</h3>
      {children}
    </div>
  );

  const NumberDisplay = ({ value, label, color = "bg-black text-green-400" }) => (
    <div className="flex flex-col items-center mb-2">
      <div className={`${color} px-3 py-1 rounded font-mono text-lg border-2`}>
        {value}
      </div>
      <span className="text-xs mt-1">{label}</span>
    </div>
  );

  return (
    <div className="flex flex-col w-full h-screen bg-gray-300 overflow-hidden">
      {/* Header */}
      <div className="bg-blue-600 text-white p-4 text-center">
        <h1 className="text-2xl font-bold">IA1 THE ENTRY AREA LOGIST</h1>
        <div className="flex justify-between items-center mt-2">
          <div className="text-sm">192.168.100.178</div>
          <div className="text-sm">2025/1/18 11:22:37</div>
        </div>
      </div>

      <div className="flex flex-1 p-4 gap-4">
        {/* Left Control Panel */}
        <div className="w-64 space-y-4">
          <ControlPanel title="SYSTEM CONTROL">
            <div className="space-y-2">
              <StatusButton status="start" className="w-full">START</StatusButton>
              <StatusButton status="pause" className="w-full">PAUSE</StatusButton>
              <NumberDisplay value="1001" label="ID" />
              <NumberDisplay value="+28623" label="COUNT" color="bg-black text-yellow-400" />
            </div>
          </ControlPanel>

          <ControlPanel title="AUTO CONTROL">
            <div className="space-y-2">
              <StatusButton status="active" className="w-full">AUTOMATIC</StatusButton>
              <div className="flex gap-1">
                <button className="flex-1 bg-green-500 text-white px-2 py-1 rounded text-xs">+0</button>
                <button className="flex-1 bg-green-500 text-white px-2 py-1 rounded text-xs">+0</button>
              </div>
              <NumberDisplay value="000" label="" color="bg-pink-200 text-black" />
              <StatusButton className="w-full">CONFIRM</StatusButton>
            </div>
          </ControlPanel>

          <ControlPanel title="MANUAL CONTROL">
            <div className="space-y-2">
              <div className="flex gap-1">
                <StatusButton status="start" className="flex-1">START</StatusButton>
                <StatusButton status="pause" className="flex-1">PAUSE</StatusButton>
              </div>
              <div className="flex gap-1">
                <button className="flex-1 bg-green-500 text-white px-2 py-1 rounded text-xs">+0</button>
                <button className="flex-1 bg-green-500 text-white px-2 py-1 rounded text-xs">+0</button>
              </div>
              <NumberDisplay value="000" label="" color="bg-pink-200 text-black" />
              <StatusButton className="w-full">CONFIRM</StatusButton>
            </div>
          </ControlPanel>

          <ControlPanel title="AUTOMATIC">
            <div className="space-y-2">
              <StatusButton status="start" className="w-full">START</StatusButton>
              <StatusButton status="pause" className="w-full">PAUSE</StatusButton>
              <div className="flex gap-1">
                <NumberDisplay value="1212" label="" color="bg-black text-white" />
                <NumberDisplay value="1213" label="" color="bg-green-500 text-white" />
              </div>
            </div>
          </ControlPanel>
        </div>

        {/* Main Grid Area */}
        <div className="flex-1 bg-white border-2 border-gray-400 p-4 overflow-auto">
          <div className="grid grid-cols-30 gap-1 mb-4" style={{ gridTemplateColumns: 'repeat(30, minmax(0, 1fr))' }}>
            {gridData.map((row, rowIndex) =>
              row.map((item, colIndex) => (
                <GridCell key={`${rowIndex}-${colIndex}`} item={item} index={colIndex} />
              ))
            )}
          </div>
          
          {/* Conveyor Direction Indicators */}
          <div className="flex justify-center mt-4">
            <svg width="600" height="100" className="border">
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="blue" />
                </marker>
              </defs>
              
              {/* Conveyor lines with arrows */}
              <line x1="50" y1="25" x2="550" y2="25" stroke="blue" strokeWidth="3" markerEnd="url(#arrowhead)" />
              <line x1="50" y1="50" x2="550" y2="50" stroke="blue" strokeWidth="3" markerEnd="url(#arrowhead)" />
              <line x1="50" y1="75" x2="550" y2="75" stroke="blue" strokeWidth="3" markerEnd="url(#arrowhead)" />
              
              <text x="300" y="15" textAnchor="middle" className="text-xs font-bold">CONVEYOR DIRECTION</text>
            </svg>
          </div>

          {/* Bottom Control Buttons */}
          <div className="flex justify-center gap-4 mt-4">
            <StatusButton className="px-6 py-3">RETURN</StatusButton>
            <StatusButton status="active" className="px-6 py-3">CGL1</StatusButton>
            <StatusButton status="active" className="px-6 py-3">CGL2</StatusButton>
            <StatusButton status="active" className="px-6 py-3">CGL3</StatusButton>
            <StatusButton className="px-6 py-3 bg-yellow-500 text-white">EXIT</StatusButton>
          </div>
        </div>

        {/* Right Status Panel */}
        <div className="w-64 space-y-4">
          <ControlPanel title="STATUS INDICATORS">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs">CGL1_PX</span>
                <div className="w-16 h-4 bg-yellow-400 border border-gray-600"></div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs">CGL2_PX</span>
                <div className="w-16 h-4 bg-yellow-400 border border-gray-600"></div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs">CGL3_PX</span>
                <div className="w-16 h-4 bg-yellow-400 border border-gray-600"></div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs">CGL4_PX</span>
                <div className="w-16 h-4 bg-yellow-400 border border-gray-600"></div>
              </div>
            </div>
          </ControlPanel>

          <ControlPanel title="TRAFFIC LIGHTS">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-red-500 rounded-full border-2 border-gray-600"></div>
                <span className="text-xs">N011_CAR</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-red-500 rounded-full border-2 border-gray-600"></div>
                <span className="text-xs">N012_CAR</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-blue-500 rounded-full border-2 border-gray-600"></div>
                <span className="text-xs">N013_CAR</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-red-500 rounded-full border-2 border-gray-600"></div>
                <span className="text-xs">N014_MB_CAR</span>
              </div>
            </div>
          </ControlPanel>

          <ControlPanel title="FINAL CONTROL">
            <div className="space-y-2">
              <StatusButton status="start" className="w-full">START</StatusButton>
              <StatusButton status="pause" className="w-full">PAUSE</StatusButton>
              <NumberDisplay value="000" label="" color="bg-pink-200 text-black" />
              <NumberDisplay value="1216" label="" color="bg-green-500 text-white" />
              <NumberDisplay value="1217" label="" color="bg-black text-white" />
            </div>
          </ControlPanel>

          <ControlPanel title="SYSTEM STATUS">
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-xs">REQ_IN</span>
                <div className="w-4 h-4 bg-red-500"></div>
              </div>
              <div className="flex justify-between">
                <span className="text-xs">EXE_IN</span>
                <div className="w-4 h-4 bg-red-500"></div>
              </div>
              <div className="flex justify-between">
                <span className="text-xs">LEAVE_IN</span>
                <div className="w-4 h-4 bg-red-500"></div>
              </div>
              <div className="flex justify-between">
                <span className="text-xs">EMERGENCY_IN</span>
                <div className="w-4 h-4 bg-green-500"></div>
              </div>
            </div>
          </ControlPanel>
        </div>
      </div>
    </div>
  );
};

export default FrontOfLineTransferTrackingCgl;