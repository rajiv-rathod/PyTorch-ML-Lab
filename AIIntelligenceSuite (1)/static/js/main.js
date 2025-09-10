document.addEventListener('DOMContentLoaded', function() {
    // File upload preview
    const fileInput = document.getElementById('file-upload');
    const fileLabel = document.querySelector('.custom-file-label');
    
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileName = this.files[0].name;
            fileLabel.textContent = fileName;
        });
    }
    
    // Feature selection functionality
    const selectAllBtn = document.getElementById('select-all-features');
    const clearAllBtn = document.getElementById('clear-all-features');
    const featureCheckboxes = document.querySelectorAll('.feature-checkbox');
    
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', function() {
            featureCheckboxes.forEach(checkbox => {
                checkbox.checked = true;
            });
        });
    }
    
    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', function() {
            featureCheckboxes.forEach(checkbox => {
                checkbox.checked = false;
            });
        });
    }
    
    // Model type selection
    const modelTypeSelect = document.getElementById('model-type');
    const deepLearningOptions = document.getElementById('deep-learning-options');
    const traditionalMlOptions = document.getElementById('traditional-ml-options');
    
    if (modelTypeSelect) {
        modelTypeSelect.addEventListener('change', function() {
            if (this.value === 'deep_learning') {
                deepLearningOptions.classList.remove('d-none');
                traditionalMlOptions.classList.add('d-none');
            } else {
                deepLearningOptions.classList.add('d-none');
                traditionalMlOptions.classList.remove('d-none');
            }
        });
    }
    
    // Form validation
    const trainForm = document.getElementById('train-form');
    
    if (trainForm) {
        trainForm.addEventListener('submit', function(event) {
            // Check if at least one feature is selected
            const selectedFeatures = document.querySelectorAll('.feature-checkbox:checked');
            
            if (selectedFeatures.length === 0) {
                event.preventDefault();
                alert('Please select at least one feature');
                return false;
            }
            
            // Check if target column is selected
            const targetColumn = document.getElementById('target-column').value;
            
            if (!targetColumn) {
                event.preventDefault();
                alert('Please select a target column');
                return false;
            }
            
            // Check if model type is selected
            const modelType = document.getElementById('model-type').value;
            
            if (!modelType) {
                event.preventDefault();
                alert('Please select a model type');
                return false;
            }
            
            return true;
        });
    }
    
    // Prediction form validation
    const predictForm = document.getElementById('predict-form');
    
    if (predictForm) {
        predictForm.addEventListener('submit', function(event) {
            // Check if all inputs have values
            const inputs = document.querySelectorAll('#predict-form input[type="number"]');
            let valid = true;
            
            inputs.forEach(input => {
                if (input.value === '') {
                    valid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });
            
            if (!valid) {
                event.preventDefault();
                alert('Please fill in all feature values');
                return false;
            }
            
            return true;
        });
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
