const defaultIcon = L.divIcon({
  className: "",
  html: `
    <svg width="30" height="41" viewBox="0 0 30 41" xmlns="http://www.w3.org/2000/svg">
      <path d="M15 0C7 0 0 7 0 15c0 12 15 26 15 26s15-14 15-26C30 7 23 0 15 0z" fill="#E53E3E"/>
      <circle cx="15" cy="15" r="5" fill="white"/>
    </svg>
  `,
  iconSize: [30, 41],
  iconAnchor: [15, 41]
});

const selectedIcon = L.divIcon({
  className: "",
  html: `
    <svg width="60" height="82" viewBox="0 0 30 41" xmlns="http://www.w3.org/2000/svg">
      <path d="M15 0C7 0 0 7 0 15c0 12 15 26 15 26s15-14 15-26C30 7 23 0 15 0z" fill="#E53E3E"/>
      <circle cx="15" cy="15" r="5" fill="white"/>
    </svg>
  `,
  iconSize: [60, 82],
  iconAnchor: [30, 82]
});

const map = L.map('map').setView([35.6895, 139.6917], 12);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '© OpenStreetMap'
}).addTo(map);
map.zoomControl.setPosition('bottomright', );

const searchInput = document.getElementById('search-input');
const daySelect = document.getElementById('day-select');
const dayChoices = new Choices(daySelect, {
  searchEnabled: false,   // 不需要搜索
  itemSelectText: '',     // 选中时不显示"Press to select"
  shouldSort: false       // 不排序，保持原来的顺序
});
const daySelectMobile = document.getElementById('day-select-mobile');
const dayChoicesMobile = new Choices(daySelectMobile, {
  searchEnabled: false,
  itemSelectText: '',
  shouldSort: false
});

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
const drawer = document.getElementById('mobile-drawer');
const overlay = document.getElementById('drawer-overlay');
const bottomDrawer = document.getElementById('bottom-drawer');
const sidebarContentMobile = document.getElementById('sidebar-content-mobile');
const drawerHeader = bottomDrawer.querySelector('.drawer-header');

document.getElementById('menu-toggle').onclick = () => {
  const isOpen = drawer.classList.toggle('open');
  if (isOpen) {
    overlay.classList.add('open');
  } else {
    overlay.classList.remove('open');
  }
};
const sidebar = document.getElementById('sidebar');
const toggleStrip = document.getElementById('sidebar-toggle-strip');
const sidebarHeader = document.getElementById('sidebar-header');

const roomInput = document.getElementById('room-id');
const saveBtn = document.getElementById('save-to-room');
const loadBtn = document.getElementById('load-from-room');

const layers = {};
const routes = {};
const selectedForRoute = [];

let data = [];
let currentDay = 1;
let pendingCoords = null;
let pendingRoute = null;
let selectedMarker = null;
let selectedRoute = null;
let selectedRoutes = new Set();
let selectedPlaces = new Set();
let selectedEntry = null;
let popupUidCounter = 0;
let startY = 0;
let currentY = 0;
let isDragging = false;

const resultsDiv = document.getElementById('search-results');

const autocompleteService = new google.maps.places.AutocompleteService();
const placesService = new google.maps.places.PlacesService(document.createElement('div'));
let searchTimeout = null;

searchInput.addEventListener('input', () => {
  const query = searchInput.value.trim();
  if (searchTimeout) clearTimeout(searchTimeout);

  if (!query) {
    resultsDiv.innerHTML = '';
    return;
  }

  searchTimeout = setTimeout(() => {
    autocompleteService.getPlacePredictions(
      {
        input: query,
        componentRestrictions: { country: 'jp' },
        language: 'zh-CN'
      },
      (predictions, status) => {
        resultsDiv.innerHTML = '';

        if (status !== google.maps.places.PlacesServiceStatus.OK || !predictions) {
          resultsDiv.innerHTML = '<div>No results found.</div>';
          return;
        }

        predictions.forEach(pred => {
          const div = document.createElement('div');
          div.innerHTML = highlightMatch(pred.description, query);
          div.onclick = () => {
            getPlaceDetails(pred.place_id, pred.description);
          };
          resultsDiv.appendChild(div);
        });
      }
    );
  }, 300);
});

function getPlaceDetails(placeId, description) {
  placesService.getDetails(
    { placeId, language: 'zh-CN' },
    (place, status) => {
      if (status !== google.maps.places.PlacesServiceStatus.OK || !place.geometry) {
        alert('无法获取地点详情');
        return;
      }

      const latlng = [
        place.geometry.location.lat(),
        place.geometry.location.lng()
      ];
      map.setView(latlng, 16);
      openModal(latlng, description);
      resultsDiv.innerHTML = '';
      searchInput.value = '';
    }
  );
}

function generatePopupHtml({ icon = '', title = '', notes = '', showDelete = true, routeId = null, coords = null }) {
  const idAttr = routeId !== null ? `data-route-id="${routeId}"` : '';
  const coordStr = coords ? `${coords[0]}_${coords[1]}` : '';

  return `
    <div class="popup-card">
      <div class="popup-header">
        <span class="popup-icon">${icon || '🛣️'}</span>
        <span class="popup-title">${title || '(Untitled)'}</span>
      </div>
      <div class="popup-notes">${notes || '<em>No notes</em>'}</div>
      <div class="popup-actions">
        <button class="popup-btn edit-btn"
          ${routeId !== null ? `onclick="editRoute('${routeId}')"` : `onclick="editMarker('${coordStr}')"`}>
          Edit
        </button>
        ${showDelete ? `
          <button class="popup-btn delete-btn"
            ${routeId !== null ? `onclick="deleteRoute('${routeId}')"` : `onclick="deleteMarker('${coordStr}')"`}>
            Delete
          </button>
        ` : ''}
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
  const markerDays = data
    .filter(e => !e.deleted)
    .map(e => Number(e.day))
    .filter(d => !isNaN(d) && d !== 0);

  const routeDays = Object.keys(routes)
    .map(d => Number(d))
    .filter(d => d !== 0);

  const allDays = [...new Set([...markerDays, ...routeDays])].sort((a, b) => a - b);

  const currentExists = allDays.includes(currentDay);
  const selectedValue = currentExists ? String(currentDay) : "0";

  // 桌面端
  dayChoices.clearChoices();
  dayChoices.setChoices(
    [
      { value: "0", label: "All Days", selected: selectedValue === "0" },
      ...allDays.map(day => ({
        value: String(day),
        label: `Day ${day}`,
        selected: String(day) === selectedValue
      }))
    ],
    'value',
    'label',
    false
  );

  // 移动端
  dayChoicesMobile.clearChoices();
  dayChoicesMobile.setChoices(
    [
      { value: "0", label: "All Days", selected: selectedValue === "0" },
      ...allDays.map(day => ({
        value: String(day),
        label: `Day ${day}`,
        selected: String(day) === selectedValue
      }))
    ],
    'value',
    'label',
    false
  );

  // 更新currentDay
  currentDay = parseInt(selectedValue);
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
  const sidebarMobile = document.getElementById('sidebar-content-mobile');

  sidebar.innerHTML = '';
  sidebarMobile.innerHTML = '';

  // MARKERS
  data
    .filter(e => !e.deleted && (currentDay === 0 || e.day === currentDay))
    .forEach(entry => {
      const key = `${entry.coords[0]}_${entry.coords[1]}`;
      // 创建桌面元素
      const el = document.createElement('div');
      el.className = 'sidebar-entry';
      if (selectedPlaces.has(key)) el.classList.add('selected');

      el.innerHTML = `
        <strong>${entry.icon || '📍'} ${entry.name || '(Unnamed)'}</strong>
        <div>${entry.notes || ''}</div>
        ${currentDay === 0 ? `<div><small style="opacity:0.6;">Day ${entry.day || '(Undecided)'}</small></div>` : ''}
      `;

      el.onclick = () => {
        if (selectedPlaces.has(key)) {
          selectedPlaces.delete(key);
        } else {
          selectedPlaces.add(key);
          if (entry.coords) {
            map.setView(entry.coords, 16);
          }
        }
        updateSidebar();
        updateBatchToolbar();
      };

      sidebar.appendChild(el);

      // 创建移动端元素
      const elMobile = el.cloneNode(true);
      elMobile.onclick = () => {
        if (selectedPlaces.has(key)) {
          selectedPlaces.delete(key);
        } else {
          selectedPlaces.add(key);
          if (entry.coords) {
            map.setView(entry.coords, 16);
          }
          if (window.innerWidth <= 768) {
            bottomDrawer.classList.remove('open');
          }
        }
        updateSidebar();
        updateBatchToolbar();
      };

      sidebarMobile.appendChild(elMobile);
    });

  // ROUTES
  Object.entries(routes).forEach(([day, group]) => {
    if (currentDay !== 0 && parseInt(day) !== currentDay) return;

    group.eachLayer(polyline => {
      const id = polyline._leaflet_id;
      const popup = polyline.getPopup()?.getContent() || '';
      const textOnly = popup.replace(/<[^>]+>/g, '');
      const isSelected = selectedRoutes.has(id);

      // 创建桌面元素
      const el = document.createElement('div');
      el.className = 'sidebar-entry';
      if (isSelected) el.classList.add('selected');

      el.innerHTML = `
        <strong>🛣️ Route</strong>
        <div>${textOnly || ''}</div>
        ${currentDay === 0 ? `<div><small style="opacity:0.6;">Day ${day}</small></div>` : ''}
      `;

      el.onclick = () => {
        if (selectedRoutes.has(id)) {
          selectedRoutes.delete(id);
          polyline.setStyle({ color: 'blue', weight: 3 });
          polyline.getElement()?.classList.remove('polyline-selected');
        } else {
          selectedRoutes.add(id);
          polyline.setStyle({ color: 'orange', weight: 6 });
          polyline.getElement()?.classList.add('polyline-selected');
          const latlngs = polyline.getLatLngs();
          if (latlngs && latlngs.length > 0) {
            const mid = Math.floor(latlngs.length / 2);
            map.setView(latlngs[mid], 14);
          }
        }
        updateSidebar();
        updateBatchToolbar();
      };

      sidebar.appendChild(el);

      // 创建移动端元素
      const elMobile = el.cloneNode(true);
      elMobile.onclick = () => {
        if (selectedRoutes.has(id)) {
          selectedRoutes.delete(id);
          polyline.setStyle({ color: 'blue', weight: 3 });
          polyline.getElement()?.classList.remove('polyline-selected');
        } else {
          selectedRoutes.add(id);
          polyline.setStyle({ color: 'orange', weight: 6 });
          polyline.getElement()?.classList.add('polyline-selected');
          const latlngs = polyline.getLatLngs();
          if (latlngs && latlngs.length > 0) {
            const mid = Math.floor(latlngs.length / 2);
            map.setView(latlngs[mid], 14);
          }
          if (window.innerWidth <= 768) {
            bottomDrawer.classList.remove('open');
          }
        }
        updateSidebar();
        updateBatchToolbar();
      };

      sidebarMobile.appendChild(elMobile);
    });
  });
}

drawerHeader.addEventListener('click', () => {
  bottomDrawer.classList.toggle("open");
});

function updateBatchToolbar() {
  const toolbar = document.getElementById('batch-toolbar');
  const createRouteBtn = document.getElementById('batch-create-route');
  const count = selectedPlaces.size + selectedRoutes.size;

  if (count < 2) {
    toolbar.classList.add('hidden');
    createRouteBtn.style.display = 'none';
    return;
  }

  document.getElementById('batch-count').textContent = `${count} items selected`;
  toolbar.classList.remove('hidden');

  // 当且仅当选中了2个Marker时，显示创建Route
  if (selectedPlaces.size === 2 && selectedRoutes.size === 0) {
    createRouteBtn.style.display = 'inline-block';
  } else {
    createRouteBtn.style.display = 'none';
  }
}

document.getElementById('batch-create-route').onclick = () => {
  if (selectedPlaces.size !== 2) {
    alert('Please select exactly 2 markers to create a route.');
    return;
  }

  // 先把两个Marker放入 selectedForRoute
  selectedForRoute.length = 0;
  const ids = Array.from(selectedPlaces);
  ids.forEach(id => {
    const [lat, lon] = id.split('_').map(Number);
    const entry = data.find(e => e.coords[0] === lat && e.coords[1] === lon && !e.deleted);
    if (entry) {
      selectedForRoute.push(entry);
    }
  });

  // 创建 pendingRoute
  const coords = selectedForRoute.map(e => e.coords);
  pendingRoute = coords;

  // 清空多选状态
  selectedPlaces.clear();
  updateBatchToolbar();
  updateSidebar();

  // 显示Route Modal
  routeModal.style.display = 'block';
};

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
    coords: entry.coords,
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
  // 移除与此Marker相关的Route
  Object.entries(routes).forEach(([day, group]) => {
    const toRemove = [];
    group.eachLayer(polyline => {
      const latlngs = polyline.getLatLngs();
      const match = latlngs.some(ll => Math.abs(ll.lat - lat) < 1e-6 && Math.abs(ll.lng - lon) < 1e-6);
      if (match) {
        toRemove.push(polyline);
        selectedRoutes.delete(polyline._leaflet_id);
      }
    });
    toRemove.forEach(line => {
      group.removeLayer(line);
    });
  });

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
    coords: entry.coords,
    showDelete: true
  });

  const marker = L.marker(entry.coords, {
    title: entry.name,
    icon: defaultIcon
  }).bindPopup(popupContent, {autoClose: false, closeOnClick: false, offset: [0, -82]})
  .on('popupopen', (e) => {
    const container = e.popup.getElement();
    if (!container) return;
  
    const editBtn = container.querySelector('[data-action="edit"]');
    const deleteBtn = container.querySelector('[data-action="delete"]');
  
    if (editBtn) {
      editBtn.onclick = () => {
        modalName.value = entry.name;
        modalNotes.value = entry.notes;
        modalIcon.value = entry.icon;
        modalDay.value = entry.day > 0 ? entry.day : '';
        pendingCoords = entry.coords;
        selectedMarker = marker;
        selectedEntry = entry;
        setMarkerSelected(marker, true);
        modal.style.display = 'block';
        marker.closePopup();
      };
    }
  
    if (deleteBtn) {
      deleteBtn.onclick = () => {
        window.deleteMarker(`${entry.coords[0]}_${entry.coords[1]}`);
      };
    }
  })
  .on('contextmenu', () => {
    handleRouteSelection(entry, marker);
  })
  .on('click', () => {
    const key = `${entry.coords[0]}_${entry.coords[1]}`;
  
    if (selectedPlaces.has(key)) {
      selectedPlaces.delete(key);
      setMarkerSelected(marker, false);
      marker.closePopup();
    } else {
      selectedPlaces.add(key);
      setMarkerSelected(marker, true);
      marker.openPopup();
    }
  
    updateSidebar();
    updateBatchToolbar();
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

  if (selectedRoute) {
    // 更新Route
    const popupContent = generatePopupHtml({
      icon: modalIcon.value,
      title: modalName.value,
      notes: modalNotes.value,
      routeId: selectedRoute._leaflet_id
    });
    selectedRoute.bindPopup(popupContent);
    selectedRoute.setStyle({ color: 'blue', weight: 3 });
    selectedRoutes.delete(selectedRoute._leaflet_id);
    selectedRoute.closePopup();
  } else if (selectedEntry) {
    // 更新Marker
    selectedEntry.name = modalName.value;
    selectedEntry.notes = modalNotes.value;
    selectedEntry.icon = modalIcon.value;
    selectedEntry.day = chosenDay;
    updateMarker(selectedEntry);

    const oldLayer = layers[selectedEntry.day];
    if (oldLayer && oldLayer !== layers[chosenDay]) {
      oldLayer.removeLayer(selectedEntry.marker);
      if (!layers[chosenDay]) {
        layers[chosenDay] = L.layerGroup().addTo(map);
      }
      layers[chosenDay].addLayer(selectedEntry.marker);
    }
  } else {
    // 新建Marker
    const newEntry = {
      name: modalName.value,
      coords: pendingCoords,
      day: chosenDay,
      notes: modalNotes.value,
      icon: modalIcon.value,
      deleted: false
    };
    data.push(newEntry);
    addMarker(newEntry);
  }

  // 关闭Modal
  modal.style.display = 'none';
  pendingCoords = null;
  if (selectedMarker) {
    resetMarkerIcon(selectedMarker);
    selectedMarker = null;
  }
  selectedEntry = null;
  selectedRoute = null;
  renderDayOptions();
  updateSidebar();
  updateBatchToolbar();
};

modalCancel.onclick = function () {
  if (selectedMarker) {
    resetMarkerIcon(selectedMarker);
    selectedMarker = null;
  }
  modal.style.display = 'none';
  pendingCoords = null;
  selectedEntry = null;
};

routeSave.onclick = function () {
  const name = routeName.value.trim();
  const icon = routeIcon.value.trim();
  const notes = routeNotes.value.trim();

  drawRoute(currentDay, pendingRoute[0], pendingRoute[1], { name, icon, notes });
  renderDayOptions();

  selectedForRoute.forEach(e => {
    setMarkerSelected(e.marker, false);
    e.marker.closePopup();
  });
  selectedForRoute.length = 0;
  routeModal.style.display = 'none';
  // 同时清理多选状态
  selectedPlaces.forEach(key => {
    const [lat, lon] = key.split('_').map(Number);
    const entry = data.find(e => e.coords[0] === lat && e.coords[1] === lon && !e.deleted);
    if (entry && entry.marker) {
      setMarkerSelected(entry.marker, false);
      entry.marker.closePopup();
    }
  });
  selectedPlaces.clear();
  updateSidebar();
  updateBatchToolbar();
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
  if (!routes[day]) routes[day] = L.layerGroup().addTo(map);
  routes[day].addLayer(polyline);
  const id = polyline._leaflet_id;

  const updatePopup = () => {
    const popupContent = generatePopupHtml({
      icon: currentIcon,
      title: currentName,
      notes: currentNote,
      routeId: id
    });
    polyline.bindPopup(popupContent);
  };

  updatePopup();

  polyline.on('click', (e) => {
    L.DomEvent.stopPropagation(e);
    const id = polyline._leaflet_id;
  
    if (selectedRoutes.has(id)) {
      selectedRoutes.delete(id);
      polyline.setStyle({ color: 'blue', weight: 3 });
    } else {
      selectedRoutes.add(id);
      polyline.setStyle({ color: 'orange', weight: 6 });
    }
  
    updateSidebar();
    updateBatchToolbar();
  });

  polyline.on('popupopen', (e) => {
    const container = e.popup.getElement();
    if (!container) return;
  
    const editBtn = container.querySelector('[data-action="edit"]');
    const deleteBtn = container.querySelector('[data-action="delete"]');
  
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
  selectedEntry = null; // 新建时清空
  modalName.value = defaultName;
  modalNotes.value = '';
  modalIcon.value = '';
  modalDay.value = '';
  modal.style.display = 'block';
}

function highlightMatch(text, query) {
  const regex = new RegExp(`(${query})`, 'gi');
  return text.replace(regex, '<strong>$1</strong>');
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

function highlightMarker(marker) {
  marker.getElement()?.classList.add('selected');
}

function resetMarkerIcon(marker) {
  marker.getElement()?.classList.remove('selected');
}

function setMarkerSelected(marker, selected) {
  marker.setIcon(selected ? selectedIcon : defaultIcon);
}

map.on('click', function (e) {
  if (e.originalEvent._stoppedByLeaflet) return;

  selectedRoutes.forEach(id => {
    const polyline = getPolylineById(id);
    if (polyline) {
      polyline.setStyle({ color: 'blue', weight: 3 });
    }
  });
  selectedRoutes.clear();

  selectedPlaces.forEach(key => {
    const [lat, lon] = key.split('_').map(Number);
    const entry = data.find(e => e.coords[0] === lat && e.coords[1] === lon && !e.deleted);
    if (entry && entry.marker) {
      setMarkerSelected(entry.marker, false);
      entry.marker.closePopup();
    }
  });
  selectedPlaces.clear();


  updateSidebar();
  updateBatchToolbar();
  // 点击空白收起底部抽屉
  if (window.innerWidth <= 768) {
    bottomDrawer.classList.remove("open");
  }
});

daySelect.addEventListener('change', () => {
  const selected = parseInt(daySelect.value);
  if (!isNaN(selected)) {
    currentDay = selected;
    switchDay(currentDay);
  }
});

renderDayOptions();
daySelect.dispatchEvent(new Event('change'));

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

saveBtn.onclick = async () => {
  const roomId = roomInput.value.trim();
  if (!roomId) {
    alert('Please enter a Room ID.');
    return;
  }

  const markers = data.map(e => {
    const { marker, ...plain } = e;
    return plain;
  });

  const routeList = [];
  Object.keys(routes).forEach(day => {
    routes[day].eachLayer(line => {
      const latlngs = line.getLatLngs();
      routeList.push({
        day: parseInt(day),
        from: [latlngs[0].lat, latlngs[0].lng],
        to: [latlngs[1].lat, latlngs[1].lng],
        notes: line.getPopup()?.getContent()?.replace(/<[^>]+>/g, '')
      });
    });
  });

  const docRef = window.firestoreDoc(window.db, "maps", roomId);
  await window.firestoreSetDoc(docRef, { markers, routes: routeList });

  alert('Data saved to room "' + roomId + '"!');
};

loadBtn.onclick = async () => {
  const roomId = roomInput.value.trim();
  if (!roomId) {
    alert('Please enter a Room ID.');
    return;
  }

  const docRef = window.firestoreDoc(window.db, "maps", roomId);
  const snap = await window.firestoreGetDoc(docRef);
  if (snap.exists()) {
    const json = snap.data();
    loadImportedData(json);
    alert('Data loaded from room "' + roomId + '"!');
  } else {
    alert('No data found for this room.');
  }
};

window.editMarker = function(coordStr) {
  const [lat, lon] = coordStr.split('_').map(Number);
  const entry = data.find(e => e.coords[0] === lat && e.coords[1] === lon && !e.deleted);
  if (!entry) return;

  modalName.value = entry.name;
  modalNotes.value = entry.notes;
  modalIcon.value = entry.icon;
  modalDay.value = entry.day > 0 ? entry.day : '';
  pendingCoords = entry.coords;
  selectedMarker = entry.marker;
  selectedEntry = entry;
  setMarkerSelected(entry.marker, true);
  modal.style.display = 'block';
  if (entry.marker) entry.marker.closePopup();
  selectedPlaces.delete(`${entry.coords[0]}_${entry.coords[1]}`);
  setMarkerSelected(entry.marker, false);
  updateSidebar();
  updateBatchToolbar();
};

window.deleteMarker = function(coordStr) {
  const [lat, lon] = coordStr.split('_').map(Number);
  const entry = data.find(e => e.coords[0] === lat && e.coords[1] === lon && !e.deleted);
  if (entry) {

    // 删除与此Marker相关的Route
    Object.entries(routes).forEach(([day, group]) => {
      const toRemove = [];
      group.eachLayer(polyline => {
        const latlngs = polyline.getLatLngs();
        const isRelated = latlngs.some(ll => {
          return Math.abs(ll.lat - lat) < 1e-6 && Math.abs(ll.lng - lon) < 1e-6;
        });
        if (isRelated) {
          toRemove.push(polyline);
          selectedRoutes.delete(polyline._leaflet_id);
        }
      });
      toRemove.forEach(polyline => {
        group.removeLayer(polyline);
      });
    });

    if (entry.marker) {
      const layer = layers[entry.day];
      if (layer) layer.removeLayer(entry.marker);
    }
    entry.deleted = true;
  }

  selectedPlaces.delete(coordStr);

  renderDayOptions();
  updateSidebar();

  if (currentDay !== 0 && data.filter(e => !e.deleted && e.day === currentDay).length === 0) {
    currentDay = 0;
  }
  switchDay(currentDay);
};

window.editRoute = function(routeId) {
  routeId = parseInt(routeId);
  let found = null;
  Object.values(routes).forEach(group => {
    group.eachLayer(layer => {
      if (layer._leaflet_id === routeId) {
        found = layer;
      }
    });
  });
  if (!found) {
    alert('Route not found.');
    return;
  }

  // 从Popup里提取当前内容
  const popup = found.getPopup()?.getContent() || '';
  const name = currentNameFromPopup(popup);
  const notes = currentNotesFromPopup(popup);
  const icon = currentIconFromPopup(popup);

  // 填入Modal
  modalName.value = name;
  modalNotes.value = notes;
  modalIcon.value = icon;
  modalDay.value = ''; // Route一般不分Day，你可以改成默认0
  selectedRoute = found;
  selectedEntry = null; // 确保不和Marker冲突
  modal.style.display = 'block';
};

window.deleteRoute = function(routeId) {
  routeId = parseInt(routeId);
  Object.entries(routes).forEach(([day, group]) => {
    group.eachLayer(line => {
      if (line._leaflet_id === routeId) {
        group.removeLayer(line);
        selectedRoutes.delete(routeId);
        updateSidebar();
      }
    });
  });
};

function currentNameFromPopup(html) {
  const div = document.createElement('div');
  div.innerHTML = html;
  const title = div.querySelector('.popup-title');
  return title ? title.textContent.trim() : '';
}

function currentNotesFromPopup(html) {
  const div = document.createElement('div');
  div.innerHTML = html;
  const notes = div.querySelector('.popup-notes');
  return notes ? notes.textContent.trim() : '';
}

function currentIconFromPopup(html) {
  const div = document.createElement('div');
  div.innerHTML = html;
  const icon = div.querySelector('.popup-icon');
  return icon ? icon.textContent.trim() : '';
}

document.getElementById('menu-toggle').onclick = () => {
  const isOpen = drawer.classList.toggle('open');
  if (isOpen) {
    overlay.classList.add('open');
  } else {
    overlay.classList.remove('open');
  }
};

overlay.onclick = () => {
  drawer.classList.remove('open');
  overlay.classList.remove('open');
};


// Save
document.getElementById('save-to-room-mobile').onclick = () => {
  const roomId = document.getElementById('room-id-mobile').value.trim();
  document.getElementById('room-id').value = roomId;
  document.getElementById('save-to-room').click();
};

// Load
document.getElementById('load-from-room-mobile').onclick = () => {
  const roomId = document.getElementById('room-id-mobile').value.trim();
  document.getElementById('room-id').value = roomId;
  document.getElementById('load-from-room').click();
};

// Day select
document.getElementById('day-select-mobile').onchange = (e) => {
  document.getElementById('day-select').value = e.target.value;
  document.getElementById('day-select').dispatchEvent(new Event('change'));
};

// 触摸开始
drawerHeader.addEventListener('touchstart', (e) => {
  startY = e.touches[0].clientY;
  currentY = startY;
  isDragging = true;
  bottomDrawer.style.transition = 'none';
});

// 触摸移动
drawerHeader.addEventListener('touchmove', (e) => {
  if (!isDragging) return;
  currentY = e.touches[0].clientY;
  const deltaY = currentY - startY;
  if (deltaY > 0) {
    // 临时跟随偏移
    bottomDrawer.style.transform = `translateY(${deltaY}px)`;
  }
});

// 触摸结束
drawerHeader.addEventListener('touchend', () => {
  if (!isDragging) return;
  isDragging = false;
  bottomDrawer.style.transition = 'transform 0.3s ease';
  bottomDrawer.style.transform = ""; // 清除行内样式

  const deltaY = currentY - startY;

  // 小于10px的触摸视为点击，不处理
  if (Math.abs(deltaY) < 10) {
    return;
  }

  if (deltaY > 100) {
    bottomDrawer.classList.remove('open');
  } else {
    bottomDrawer.classList.add('open');
  }
});
