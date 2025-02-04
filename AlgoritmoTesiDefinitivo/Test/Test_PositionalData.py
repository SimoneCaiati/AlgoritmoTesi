import pytest
import numpy as np
import pandas as pd
import os

from PositionGetters.PositionalData import PositionalData

# Creazione di dati di test finti
@pytest.fixture
def fake_data():
    timestamp = np.linspace(0, 10, 100)  # Simuliamo 100 campioni in 10 secondi
    acc = np.zeros((100, 3))  # Nessuna accelerazione 
    orient = np.zeros((100, 3))
    sample_rate = 10  # 10 Hz
    file_index = '0'
    directory='0'
    specificPD='0'

    return timestamp, acc, orient, sample_rate, file_index, directory, specificPD

# 1.Test del costruttore (__init__)
def test_init(fake_data):
    timestamp, acc, orient, sample_rate, file_index, directory, specificPD = fake_data
    pg = PositionalData(timestamp,acc,orient,sample_rate,file_index,directory,True,specificPD)

    assert len(pg.timestamp) == len(timestamp)
    assert pg.Acc.shape == acc.shape
    assert pg.Orient.shape == orient.shape
    assert pg.sample_rate == sample_rate
    assert pg.file_index == file_index
    assert pg.directory == directory

# 2.Test della funzione identify_moving_periods: verifico che il vettore di movimento sia della stessa grandezza del vettore timestamp e che contenga solo
#   valori booleani    
def test_identify_moving_periods(fake_data):
    timestamp, acc, orient, sample_rate, file_index, directory, specificPD = fake_data
    pg = PositionalData(timestamp,acc,orient,sample_rate,file_index,directory,True,specificPD)
    
    pg.identify_moving_periods(pg.Acc)  # Esegui il metodo
    
    assert pg.is_moving.shape == (len(timestamp),)  # Deve essere un vettore della stessa lunghezza di timestamp
    assert np.all((pg.is_moving == 0) | (pg.is_moving == 1))  # Deve contenere solo 0 o 
    
# 3.Test della funzione getPosition: verifico che velocità e posizione abbiano lo stesso numero di elementi del timestamp
def test_getPositionData_output_shape(fake_data):
    timestamp, acc, orient, sample_rate, file_index, directory, specificPD = fake_data
    pg = PositionalData(timestamp,acc,orient,sample_rate,file_index,directory,True,specificPD)
    
    pg.identify_moving_periods(pg.Acc) 
    
    pg.getPositionData(pg.Acc, "TestData")

    assert pg.velocity.shape == (len(timestamp), 3)
    assert pg.position.shape == (len(timestamp), 3)
 
# 4.Test del metodo getPosition: verifico che la velocità sia uguale a [0,0,0] all'inizio del moto
def test_getPositionData_initial_velocity(fake_data):
    timestamp, acc, orient, sample_rate, file_index, directory, specificPD = fake_data
    pg = PositionalData(timestamp,acc,orient,sample_rate,file_index,directory,True,specificPD)

    pg.identify_moving_periods(pg.Acc) 
    pg.getPositionData(pg.Acc, "TestData")

    np.testing.assert_array_equal(pg.velocity[0], np.zeros(3))

# 5.Test del metodo getPosition: verifico che in assenza di accelerazione la velocità rimanga costante
def test_getPositionData_constant_velocity():
    timestamp = np.linspace(0, 10, 100)  # Simuliamo 100 campioni in 10 secondi
    acc = np.zeros((100, 3))  # Nessuna accelerazione 
    orient = np.zeros((100, 3))
    sample_rate = 10  # 10 Hz
    file_index = '0'
    directory='0'
    specificPD='0'

    pg = PositionalData(timestamp,acc,orient,sample_rate,file_index,directory,True,specificPD)
    pg.identify_moving_periods(pg.Acc) 
    pg.getPositionData(pg.Acc, "TestData")

    np.testing.assert_array_equal(pg.velocity, np.zeros((100, 3)))  # Velocità deve rimanere zero
 
    