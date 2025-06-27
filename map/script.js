const map = L.map('map').setView([35.6895, 139.6917], 12);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '© OpenStreetMap'
}).addTo(map);
map.zoomControl.setPosition('bottomright');

const daySelect = document.getElementById('day-select');
const exportBtn = document.getElementById('export-json');
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');

const modal = document.getElementById('input-modal');
const modalName = document.getElementById('modal-name');
const modalNotes = document.getElementById('modal-notes');
const modalIcon = document.getElementById('modal-icon');
const modalSave = document.getElementById('modal-save');
const modalCancel = document.getElementById('modal-cancel');
const modalDay = document.getElementById('modal-day');

const routeModal = document.getElementById('route-modal');
const routeName = document.getElementById('route-name');
const routeIcon = document.getElementById('route-icon');
const routeNotes = document.getElementById('route-notes');
const routeSave = document.getElementById('route-save');
const routeCancel = document.getElementById('route-cancel');
const drawer = document.getElementById('bottom-drawer');
const sidebar = document.getElementById('sidebar');
const toggleStrip = document.getElementById('sidebar-toggle-strip');
const sidebarHeader = document.getElementById('sidebar-header');


let data = [];
let currentDay = 1;
const layers = {};
const routes = {};
const selectedForRoute = [];
let pendingCoords = null;
let pendingRoute = null;
let selectedMarker = null;
let selectedRoutes = new Set();
let selectedPlaces = new Set();

function generatePopupHtml({ icon = '', title = '', notes = '', onEditId, onDeleteId, showDelete = true }) {
  return `
    <div class="popup-card">
      <div class="popup-header">
        <span class="popup-icon">${icon || '📍'}</span>
        <span class="popup-title">${title || '(Untitled)'}</span>
      </div>
      <div class="popup-notes">${notes || '<em>No notes</em>'}</div>
      <div class="popup-actions">
        <button class="popup-btn edit-btn" id="${onEditId}">Edit</button>
        ${showDelete ? `<button class="popup-btn delete-btn" id="${onDeleteId}">Delete</button>` : ''}
      </div>
    </div>`;
}

function highlightMarker(marker) {
  marker.getElement()?.classList.add('highlighted');
}

function resetMarkerIcon(marker) {
  marker.getElement()?.classList.remove('highlighted');
}

function renderDayOptions() {
  const validDays = data
    .filter(e => !e.deleted)
    .map(e => Number(e.day))
    .filter(d => !isNaN(d));

  const days = [...new Set(validDays)].sort((a, b) => a - b);
  daySelect.innerHTML = '';

  const allOpt = document.createElement('option');
  allOpt.value = 0;
  allOpt.textContent = 'All Days';
  daySelect.appendChild(allOpt);

  days.forEach(day => {
    if (day === 0) return;
    const opt = document.createElement('option');
    opt.value = day;
    opt.textContent = `Day ${day}`;
    daySelect.appendChild(opt);
  });

  if (!days.includes(currentDay)) {
    currentDay = 0;
  }

  daySelect.value = currentDay;
}

function switchDay(day) {
  Object.values(layers).forEach(layer => map.removeLayer(layer));
  Object.values(routes).forEach(route => map.removeLayer(route));

  if (day === 0) {
    Object.values(layers).forEach(layer => map.addLayer(layer));
    Object.values(routes).forEach(route => map.addLayer(route));
  } else {
    if (layers[day]) map.addLayer(layers[day]);
    if (routes[day]) map.addLayer(routes[day]);
  }

  updateSidebar();
}

function updateSidebar() {
  const sidebar = document.getElementById('sidebar-content');
  sidebar.innerHTML = '';

  data
    .filter(e => !e.deleted && (currentDay === 0 || e.day === currentDay))
    .forEach(entry => {
      const el = document.createElement('div');
      el.className = 'sidebar-entry';
      el.innerHTML = `
        <strong>${entry.icon || '📍'} ${entry.name || '(Unnamed)'}</strong>
        <div>${entry.notes || ''}</div>
      `;

      const key = `${entry.coords[0]}_${entry.coords[1]}`;
      if (selectedPlaces.has(key)) {
        el.classList.add('selected');
      }

      if (currentDay === 0) {
        const dayLabel = entry.day > 0 ? `Day ${entry.day}` : '(Undecided)';
        el.innerHTML += `<div><small style="opacity: 0.6;">${dayLabel}</small></div>`;
      }

      el.onclick = () => {
        if (selectedPlaces.has(key)) {
          selectedPlaces.delete(key);
        } else {
          selectedPlaces.add(key);
        }
        updateSidebar();
      };
      sidebar.appendChild(el);
    });

  updateBatchToolbar?.(); // if defined
}

function updateBatchToolbar() {
  const toolbar = document.getElementById('batch-toolbar');
  const count = selectedPlaces.size + selectedRoutes.size;
  if (count < 2) {
    toolbar.classList.add('hidden');
    return;
  }
  document.getElementById('batch-count').textContent = `${count} items selected`;
  toolbar.classList.remove('hidden');
}

document.getElementById('batch-delete').onclick = () => {
  for (const id of selectedPlaces) {
    window.deleteMarker(id);
  }
  for (const id of selectedRoutes) {
    for (const day in routes) {
      routes[day].eachLayer(line => {
        if (line._leaflet_id === parseInt(id)) {
          routes[day].removeLayer(line);
        }
      });
    }
  }
  selectedPlaces.clear();
  selectedRoutes.clear();
  renderDayOptions();
  switchDay(currentDay);
  updateBatchToolbar();
};

function updateMarker(entry) {
  const popupId = `popup_${entry.coords[0]}_${entry.coords[1]}`;
  const popupContent = generatePopupHtml({
    icon: entry.icon,
    title: entry.name,
    notes: entry.notes,
    onEditId: `edit-${popupId}`,
    onDeleteId: `delete-${popupId}`,
    showDelete: true
  });
  entry.marker.setPopupContent(popupContent);
  entry.marker.options.title = entry.name;
}

window.deleteMarker = function (id) {
  const [lat, lon] = id.split('_').map(Number);
  const entry = data.find(e => e.coords[0] === lat && e.coords[1] === lon && !e.deleted);
  if (entry) {
    layers[entry.day]?.removeLayer(entry.marker);
    entry.deleted = true;
  }
  selectedPlaces.delete(id);

  renderDayOptions();
  updateSidebar();

  const remaining = data.filter(e => !e.deleted && e.day === currentDay);
  if (remaining.length === 0 && currentDay !== 0) {
    currentDay = 0;
  }
  switchDay(currentDay);
};

function handleRouteSelection(entry, marker) {
  if (selectedForRoute.includes(entry)) return;

  highlightMarker(marker);
  selectedForRoute.push(entry);

  if (selectedForRoute.length === 2) {
    pendingRoute = [selectedForRoute[0].coords, selectedForRoute[1].coords];
    routeNotes.value = '';
    routeName.value = '';
    routeIcon.value = '';
    routeModal.style.display = 'block';
  }
}

function addMarker(entry) {
  const popupId = `popup_${entry.coords[0]}_${entry.coords[1]}`;
  const popupContent = generatePopupHtml({
    icon: entry.icon,
    title: entry.name,
    notes: entry.notes,
    onEditId: `edit-${popupId}`,
    onDeleteId: `delete-${popupId}`,
    showDelete: true
  });

  const marker = L.marker(entry.coords, {
    title: entry.name
  }).bindPopup(popupContent)
    .on('popupopen', () => {
      const editBtn = document.getElementById(`edit-${popupId}`);
      if (editBtn) {
        editBtn.onclick = () => {
          modalName.value = entry.name;
          modalNotes.value = entry.notes;
          modalIcon.value = entry.icon;
          pendingCoords = entry.coords;
          selectedMarker = marker;
          highlightMarker(marker);
          modal.style.display = 'block';
          marker.closePopup();
        };
      }
      const delBtn = document.getElementById(`delete-${popupId}`);
      if (delBtn) {
        delBtn.onclick = () => {
          window.deleteMarker(`${entry.coords[0]}_${entry.coords[1]}`);
        };
      }
    })
    .on('contextmenu', () => {
      handleRouteSelection(entry, marker);
    });

  entry.marker = marker;
  if (!layers[entry.day]) layers[entry.day] = L.layerGroup().addTo(map);
  layers[entry.day].addLayer(marker);
  updateSidebar();
}

modalSave.onclick = function () {
  let chosenDay = parseInt(modalDay.value);
  if (isNaN(chosenDay)) {
    chosenDay = 0;
  }
  if (chosenDay < 0) {
    alert("Please enter a valid day number (1 or greater, or leave blank for All Days)");
    return;
  }
  if (!pendingCoords) {
    alert("Missing location coordinates.");
    return;
  }

  const newEntry = {
    name: modalName.value,
    coords: pendingCoords,
    day: chosenDay,
    notes: modalNotes.value,
    icon: modalIcon.value,
    deleted: false
  };

  // Always push new entry — don't dedupe by coords
  data.push(newEntry);
  addMarker(newEntry);

  renderDayOptions();
  updateSidebar();

  modal.style.display = 'none';
};

modalCancel.onclick = function () {
  if (selectedMarker) {
    resetMarkerIcon(selectedMarker);
    selectedMarker = null;
  }
  modal.style.display = 'none';
};

routeSave.onclick = function () {
  const name = routeName.value.trim();
  const icon = routeIcon.value.trim();
  const notes = routeNotes.value.trim();

  drawRoute(currentDay, pendingRoute[0], pendingRoute[1], { name, icon, notes });

  selectedForRoute.forEach(e => resetMarkerIcon(e.marker));
  selectedForRoute.length = 0;
  routeModal.style.display = 'none';
};

routeCancel.onclick = function () {
  selectedForRoute.forEach(e => resetMarkerIcon(e.marker));
  selectedForRoute.length = 0;
  routeModal.style.display = 'none';
};

function drawRoute(day, from, to, { name = 'Route', icon = '', notes = '' } = {}) {
  const routeId = `route_${day}_${from[0]}_${from[1]}_${to[0]}_${to[1]}`;
  let currentNote = notes;
  let currentName = name;
  let currentIcon = icon;

  const polyline = L.polyline([from, to], { color: 'blue', weight: 3 });
  const id = polyline._leaflet_id;

  const updatePopup = () => {
    const popupContent = generatePopupHtml({
      icon: currentIcon,
      title: currentName,
      notes: currentNote,
      onEditId: `edit-${id}`,
      onDeleteId: `delete-${id}`
    });
    polyline.bindPopup(popupContent);
  };

  updatePopup();

  polyline.on('click', (e) => {
    L.DomEvent.stopPropagation(e);
    polyline.openPopup(e.latlng);

    const id = polyline._leaflet_id;

    if (selectedRoutes.has(id)) {
      selectedRoutes.delete(id);
      polyline.setStyle({ color: 'blue', weight: 3 });
      polyline.getElement()?.classList.remove('polyline-selected');
    } else {
      selectedRoutes.add(id);
      polyline.setStyle({ color: 'orange', weight: 6 });
      polyline.getElement()?.classList.add('polyline-selected');
    }
    updateSidebar();
  });

  polyline.on('popupopen', () => {
    const editBtn = document.getElementById(`edit-${id}`);
    const deleteBtn = document.getElementById(`delete-${id}`);

    if (editBtn) {
      editBtn.onclick = () => {
        const newName = prompt('Edit route name:', currentName);
        if (newName !== null) currentName = newName;

        const newIcon = prompt('Edit route icon (emoji):', currentIcon);
        if (newIcon !== null) currentIcon = newIcon;

        const newNote = prompt('Edit route notes:', currentNote);
        if (newNote !== null) currentNote = newNote;

        updatePopup();
      };
    }

    if (deleteBtn) {
      deleteBtn.onclick = () => {
        if (confirm('Delete this route?')) {
          routes[day].removeLayer(polyline);
          selectedRoutes.delete(id);
          updateSidebar();
        }
      };
    }
  });

  if (!routes[day]) routes[day] = L.layerGroup().addTo(map);
  routes[day].addLayer(polyline);
  updateSidebar();
}

function handleMapClick(e) {
  const targetTag = e.originalEvent.target.tagName;
  if (['BUTTON', 'INPUT', 'TEXTAREA', 'SPAN', 'DIV', 'SVG', 'PATH'].includes(targetTag)) {
    return; // 点击的是控件或图层，不处理
  }
  openModal([e.latlng.lat, e.latlng.lng]);
}

function openModal(latlng, defaultName = '') {
  pendingCoords = latlng;
  modalName.value = defaultName;
  modalNotes.value = '';
  modalIcon.value = '';
  modal.style.display = 'block';
}

async function searchLocation() {
  const query = searchInput.value.trim();
  if (!query) return;
  const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`;
  const response = await fetch(url);
  const results = await response.json();
  if (results.length === 0) {
    alert('No results found.');
    return;
  }
  const loc = results[0];
  const latlng = [parseFloat(loc.lat), parseFloat(loc.lon)];
  map.setView(latlng, 15);
  openModal(latlng, loc.display_name);
}

function exportData() {
  const markers = data.map(e => {
    const { marker, ...plain } = e;
    return plain;
  });

  const routeList = [];
  Object.keys(routes).forEach(day => {
    routes[day].eachLayer(line => {
      const popup = line.getPopup()?.getContent() || '';
      const textOnly = popup.replace(/<[^>]*>/g, '');
      const latlngs = line.getLatLngs();
      routeList.push({
        day: parseInt(day),
        from: [latlngs[0].lat, latlngs[0].lng],
        to: [latlngs[1].lat, latlngs[1].lng],
        notes: textOnly, // or parse HTML if you store structured data
      });
    });
  });

  const blob = new Blob([JSON.stringify({ markers, routes: routeList }, null, 2)], {
    type: 'application/json'
  });
  saveAs(blob, 'trip-plan.json');
}

function getPolylineById(id) {
  const group = routes[currentDay];
  if (!group) return null;

  let found = null;
  group.eachLayer(layer => {
    if (layer._leaflet_id === id) {
      found = layer;
    }
  });
  return found;
}

function loadImportedData(json) {
  Object.values(layers).forEach(layer => map.removeLayer(layer));
  Object.values(routes).forEach(layer => map.removeLayer(layer));
  data = [];
  selectedRoutes.clear();

  if (json.markers) {
    json.markers.forEach(entry => {
      data.push(entry);
      addMarker(entry);
    });
  }

  if (json.routes) {
    json.routes.forEach(route => {
      drawRoute(route.day, route.from, route.to, {
        notes: route.notes || '',
        name: route.name || 'Route',
        icon: route.icon || ''
      });
    });
  }

  renderDayOptions();
}

map.on('click', function (e) {
  if (e.originalEvent._stoppedByLeaflet) return;

  selectedRoutes.forEach(id => {
    const polyline = getPolylineById(id);
    if (polyline) {
      polyline.setStyle({ color: 'blue', weight: 3 });
      polyline.getElement()?.classList.remove('polyline-selected');
    }
  });
  selectedRoutes.clear();

  handleMapClick(e);
});

daySelect.addEventListener('change', () => {
  const selected = parseInt(daySelect.value);
  if (!isNaN(selected)) {
    currentDay = selected;
    switchDay(currentDay);
  }
});

exportBtn.addEventListener('click', exportData);
searchBtn.addEventListener('click', searchLocation);

renderDayOptions();
daySelect.dispatchEvent(new Event('change'));

document.getElementById('import-json-btn').onclick = () => {
  document.getElementById('import-json').click();
};

document.getElementById('import-json').addEventListener('change', async function () {
  const file = this.files[0];
  if (!file) return;

  const text = await file.text();
  try {
    const json = JSON.parse(text);
    loadImportedData(json);
  } catch (err) {
    alert('Invalid JSON file.');
  }
});

toggleStrip.onclick = () => {
  sidebar.classList.remove('collapsed');
};

sidebarHeader.onclick = () => {
  sidebar.classList.add('collapsed');
};

// Optional: touch up to open mobile drawer
drawer.addEventListener('touchstart', (e) => {
  const startY = e.touches[0].clientY;
  drawer.addEventListener('touchmove', (e2) => {
    const deltaY = e2.touches[0].clientY - startY;
    if (deltaY < -50) drawer.classList.add('open');
    if (deltaY > 50) drawer.classList.remove('open');
  }, { once: true });
});
