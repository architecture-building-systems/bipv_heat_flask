// Common JavaScript functions for BIPV Heat Dashboard
// This file contains shared functionality across multiple pages

// Global variables that may be used across pages
let allExperiments = [];
let filterRanges = {};

// Local storage keys
const SERIES_STORAGE_KEY_SINGLE = 'bipv_selected_series';
const SERIES_STORAGE_KEY_COMPARE = 'bipv_compare_selected_series';

// Default series - air temperature from comfort cube
const DEFAULT_SERIES = 'Comfort Cube_standard_effective_temperature-degrees_celsius';

// ============================================================================
// LOCAL STORAGE HELPER FUNCTIONS
// ============================================================================

function saveSelectedSeries(seriesArray, storageKey = SERIES_STORAGE_KEY_SINGLE) {
    try {
        localStorage.setItem(storageKey, JSON.stringify(seriesArray));
    } catch (e) {
        console.warn('Failed to save series selection to localStorage:', e);
    }
}

function getSavedSeries(storageKey = SERIES_STORAGE_KEY_SINGLE) {
    try {
        const saved = localStorage.getItem(storageKey);
        return saved ? JSON.parse(saved) : null;
    } catch (e) {
        console.warn('Failed to load series selection from localStorage:', e);
        return null;
    }
}

// ============================================================================
// EXPERIMENT DATA LOADING
// ============================================================================

function loadExperimentCharacteristics(callback = null) {
    $.get('/api/experiment-characteristics', function(data) {
        allExperiments = data.experiments;
        filterRanges = data.ranges;
        
        // Setup radio buttons with unique values
        setupRadioButtons();
        
        // Initial filter
        filterExperiments();
        
        // Call callback if provided
        if (callback && typeof callback === 'function') {
            callback(data);
        }
    });
}

// ============================================================================
// FILTER CONTROLS
// ============================================================================

function setupRadioButtons() {
    const params = ['G', 'T', 'A', 'Ti', 'Type'];
    
    params.forEach(param => {
        const uniqueValues = filterRanges[param].values;
        const radioGroup = document.getElementById(`${param}-radio-group`);
        
        if (!radioGroup) return; // Skip if radio group doesn't exist on this page
        
        // Add radio buttons for each unique value
        uniqueValues.forEach(value => {
            const radioDiv = document.createElement('div');
            radioDiv.className = 'form-check form-check-inline';
            
            const radioInput = document.createElement('input');
            radioInput.className = 'form-check-input';
            radioInput.type = 'radio';
            radioInput.name = `${param}-filter`;
            radioInput.id = `${param}-${value}`;
            radioInput.value = value;
            
            const radioLabel = document.createElement('label');
            radioLabel.className = 'form-check-label';
            radioLabel.setAttribute('for', `${param}-${value}`);
            radioLabel.textContent = value;
            
            radioDiv.appendChild(radioInput);
            radioDiv.appendChild(radioLabel);
            radioGroup.appendChild(radioDiv);
            
            // Add event listener
            radioInput.addEventListener('change', filterExperiments);
        });
    });
    
    // Add event listeners to "All" radio buttons
    ['G', 'T', 'A', 'Ti', 'Type'].forEach(param => {
        const allButton = document.getElementById(`${param}-all`);
        if (allButton) {
            allButton.addEventListener('change', filterExperiments);
        }
    });
}

function getSelectedFilters() {
    const filters = {};
    
    ['G', 'T', 'A', 'Ti'].forEach(param => {
        const checkedRadio = document.querySelector(`input[name="${param}-filter"]:checked`);
        if (checkedRadio && checkedRadio.value !== 'all') {
            filters[param] = parseInt(checkedRadio.value);
        } else {
            filters[param] = 'all'; // No filter applied
        }
    });
    
    // Handle Type parameter separately (it's a string, not numeric)
    const typeRadio = document.querySelector(`input[name="Type-filter"]:checked`);
    if (typeRadio && typeRadio.value !== 'all') {
        filters['Type'] = typeRadio.value; // Keep as string
    } else {
        filters['Type'] = 'all'; // No filter applied
    }
    
    return filters;
}

function filterExperiments() {
    const filters = getSelectedFilters();
    
    const filteredExperiments = allExperiments.filter(exp => {
        const chars = exp.characteristics;
        return (filters.G === 'all' || chars.G === filters.G) &&
               (filters.T === 'all' || chars.T === filters.T) &&
               (filters.A === 'all' || chars.A === filters.A) &&
               (filters.Ti === 'all' || chars.Ti === filters.Ti) &&
               (filters.Type === 'all' || chars.Type_display === filters.Type);
    });
    
    // Update filtered experiments dropdown
    const dropdown = $('#filtered-experiments');
    if (dropdown.length === 0) return; // Skip if dropdown doesn't exist on this page
    
    dropdown.empty();
    
    if (filteredExperiments.length === 0) {
        dropdown.append('<option disabled>No experiments match filters</option>');
        $('#download-button').prop('disabled', true);
    } else {
        filteredExperiments.forEach(exp => {
            const chars = exp.characteristics;
            const label = `${exp.display_name} (G:${chars.G}, T:${chars.T}, A:${chars.A}, Ti:${chars.Ti})`;
            dropdown.append($('<option>', {
                value: exp.code,
                text: label
            }));
        });
        
        // Page-specific logic for auto-selection
        if (typeof handleFilteredExperimentsUpdate === 'function') {
            handleFilteredExperimentsUpdate(filteredExperiments);
        }
        
        $('#download-button').prop('disabled', false);
    }
}

// ============================================================================
// SERIES SELECTION
// ============================================================================

function selectBestSeries(availableSeries, storageKey = SERIES_STORAGE_KEY_SINGLE) {
    const seriesDropdown = $('#series-dropdown');
    const availableValues = availableSeries.map(s => s.value);
    
    // Get previously saved series
    const savedSeries = getSavedSeries(storageKey);
    
    let seriesToSelect = [];
    
    if (savedSeries && savedSeries.length > 0) {
        // Try to restore saved series - keep only those available in this experiment
        seriesToSelect = savedSeries.filter(series => availableValues.includes(series));
        
        if (seriesToSelect.length === 0) {
            console.log('Previously selected series not available in this experiment, falling back to default');
        }
    }
    
    // If no saved series could be restored, use default
    if (seriesToSelect.length === 0) {
        if (availableValues.includes(DEFAULT_SERIES)) {
            seriesToSelect = [DEFAULT_SERIES];
            console.log('Using default series:', DEFAULT_SERIES);
        } else {
            // If default not available, use first available series
            seriesToSelect = [availableSeries[0]?.value];
            console.log('Default series not available, using first available:', seriesToSelect[0]);
        }
    }
    
    // Update dropdown and current selection
    seriesDropdown.val(seriesToSelect);
    
    return seriesToSelect.filter(s => s); // Remove any null/undefined values
}

// ============================================================================
// LEGEND AND VISUALIZATION
// ============================================================================

function updateCustomLegend(legendData) {
    const legendContainer = $('#custom-legend');
    if (legendContainer.length === 0) return; // Skip if legend container doesn't exist
    
    legendContainer.empty();
    
    if (legendData.length === 0) {
        legendContainer.html('<p class="text-muted small">No series selected</p>');
        return;
    }
    
    legendData.forEach((item, index) => {
        // Use color from the legend data if available, otherwise fall back to default
        const color = item.color || getTraceColor(index);
        const lineStyle = item.style === 'dashed' ? 'dashed' : 'solid';
        
        const legendItem = $(`
            <div class="legend-item mb-2">
                <div class="d-flex align-items-center">
                    <div class="legend-line me-2" style="
                        width: 20px; 
                        height: 2px; 
                        background-color: ${color}; 
                        border: ${lineStyle === 'dashed' ? '1px dashed ' + color : 'none'};
                        background: ${lineStyle === 'dashed' ? 'none' : color};
                    "></div>
                    <div class="legend-text">
                        <div class="legend-name" style="font-size: 11px; font-weight: 600;">${item.name}</div>
                        <div class="legend-unit text-muted" style="font-size: 10px;">${item.unit}</div>
                    </div>
                </div>
            </div>
        `);
        
        legendContainer.append(legendItem);
    });
}

function getTraceColor(index) {
    // Plotly default color sequence
    const colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ];
    return colors[index % colors.length];
}

// ============================================================================
// HOME PAGE SLIDESHOW FUNCTIONALITY
// ============================================================================

let slideshowData = {};

function initializeSlideshows() {
    const slideshows = document.querySelectorAll('.facility-slideshow');
    
    slideshows.forEach(slideshow => {
        const folder = slideshow.dataset.folder;
        slideshowData[folder] = {
            currentSlide: 0,
            images: [],
            element: slideshow
        };
        
        loadImagesForSlideshow(folder);
    });
}

function loadImagesForSlideshow(folder) {
    $.get(`/api/facility-images/${folder}`, function(images) {
        if (images && images.length > 0) {
            slideshowData[folder].images = images;
            renderSlideshow(folder);
        } else {
            // Show placeholder if no images
            showPlaceholder(folder);
        }
    }).fail(function() {
        showPlaceholder(folder);
    });
}

function renderSlideshow(folder) {
    const data = slideshowData[folder];
    const slideshow = data.element;
    const imagesContainer = slideshow.querySelector('.slideshow-images');
    const dotsContainer = slideshow.querySelector('.slideshow-dots');
    
    // Clear existing content
    imagesContainer.innerHTML = '';
    dotsContainer.innerHTML = '';
    
    // Add images
    data.images.forEach((image, index) => {
        const img = document.createElement('img');
        img.src = `/static/images/facilities/${folder}/${image}`;
        img.alt = `${folder} image ${index + 1}`;
        if (index === 0) img.classList.add('active');
        imagesContainer.appendChild(img);
        
        // Add dot
        const dot = document.createElement('span');
        dot.classList.add('slideshow-dot');
        if (index === 0) dot.classList.add('active');
        dot.onclick = () => goToSlide(folder, index);
        dotsContainer.appendChild(dot);
    });
    
    // Hide navigation if only one image
    const prevBtn = slideshow.querySelector('.prev');
    const nextBtn = slideshow.querySelector('.next');
    if (data.images.length <= 1) {
        prevBtn.style.display = 'none';
        nextBtn.style.display = 'none';
        dotsContainer.style.display = 'none';
    }
}

function showPlaceholder(folder) {
    const slideshow = slideshowData[folder].element;
    const imagesContainer = slideshow.querySelector('.slideshow-images');
    const dotsContainer = slideshow.querySelector('.slideshow-dots');
    
    imagesContainer.innerHTML = `
        <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background-color: #f8f9fa; color: #6c757d;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">📷</div>
                <div style="font-size: 10px;">No images available</div>
            </div>
        </div>
    `;
    dotsContainer.innerHTML = '';
    
    // Hide navigation buttons
    slideshow.querySelector('.prev').style.display = 'none';
    slideshow.querySelector('.next').style.display = 'none';
}

function changeSlide(button, direction) {
    const slideshow = button.closest('.facility-slideshow');
    const folder = slideshow.dataset.folder;
    const data = slideshowData[folder];
    
    if (!data || data.images.length <= 1) return;
    
    data.currentSlide = (data.currentSlide + direction + data.images.length) % data.images.length;
    updateSlideDisplay(folder);
}

function goToSlide(folder, slideIndex) {
    const data = slideshowData[folder];
    if (!data || data.images.length <= 1) return;
    
    data.currentSlide = slideIndex;
    updateSlideDisplay(folder);
}

function updateSlideDisplay(folder) {
    const data = slideshowData[folder];
    const slideshow = data.element;
    
    // Update images
    const images = slideshow.querySelectorAll('.slideshow-images img');
    images.forEach((img, index) => {
        img.classList.toggle('active', index === data.currentSlide);
    });
    
    // Update dots
    const dots = slideshow.querySelectorAll('.slideshow-dot');
    dots.forEach((dot, index) => {
        dot.classList.toggle('active', index === data.currentSlide);
    });
}
