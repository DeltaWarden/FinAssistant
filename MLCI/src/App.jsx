import React, { useState, useEffect } from 'react';
import { 
  Box, Button, TextField, Typography, Paper, 
  Snackbar, Switch, ButtonGroup, Container, 
  Card, CardContent, Tabs, Tab, AppBar 
} from '@mui/material';
import { DataGrid, gridClasses } from '@mui/x-data-grid';
import Alert from '@mui/material/Alert';
import { grey } from '@mui/material/colors';
import axios from 'axios';
import { useNavigate } from 'react-router';
import { jwtDecode } from 'jwt-decode';
import styles from './App.module.css';

const API_URL = 'http://localhost:8000/api';
const api = axios.create({
  baseURL: API_URL,
  timeout: 5000,
});


api.interceptors.request.use(config => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, error => {
  return Promise.reject(error);
});

const columns = [
  { field: 'id', headerName: 'ID', width: 70 },
  { field: 'category', headerName: 'Категория', width: 150 },
  { 
    field: 'withdrawal_amount', 
    headerName: 'Сумма снятия', 
    width: 150,
    valueFormatter: (params) => {
      const value = params.value;
      if (value == null || value === 0) return '0.00 ₽'; 
      return `${parseFloat(value).toFixed(2)} ₽`;  
    }
  },
  { 
    field: 'deposit_amount', 
    headerName: 'Сумма пополнения', 
    width: 150,
    valueFormatter: (params) => {
      const value = params.value;
      if (value == null || value === 0) return '0.00 ₽'; 
      return `${parseFloat(value).toFixed(2)} ₽`;  
    }
  },
  { 
    field: 'balance', 
    headerName: 'Баланс', 
    width: 150,
    valueFormatter: (params) => {
      const value = params.value;
      if (value == null) return '0.00 ₽'; 
      return `${parseFloat(value).toFixed(2)} ₽`;  
    }
  },
  { 
    field: 'date', 
    headerName: 'Дата', 
    width: 180,
    valueFormatter: (params) => {
      const value = params.value;
      if (!value) return ''; 
      return new Date(value).toLocaleString(); 
    }
  },
  { field: 'ref_no', headerName: 'Номер транзакции', width: 180 },
];

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [authData, setAuthData] = useState({
    username: '',
    email: '',
    password: '',
    password2: ''
  });
  const [loginData, setLoginData] = useState({
    email: '',
    password: ''
  });
  const [withdrawalAmount, setWithdrawalAmount] = useState(0);  
  const [depositAmount, setDepositAmount] = useState(0); 
  const [user, setUser] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [filteredTransactions, setFilteredTransactions] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('Все');
  const [debugMode, setDebugMode] = useState(false);
  const [newTransaction, setNewTransaction] = useState({
    category: '',
    withdrawal_amount: '',
    deposit_amount: '',
    description: '',
    operation_type: 'withdrawal'
  });
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });

  const navigate = useNavigate();

  
  useEffect(() => {
    setFilteredTransactions(
      selectedCategory === 'Все' ? transactions : transactions.filter(t => t.category === selectedCategory)
    );
  }, [selectedCategory, transactions]);

  const fetchTransactions = async () => {
    try {
      const response = await api.get('/transactions/');
      const formattedData = response.data.map(item => ({
        ...item,
        withdrawal_amount: item.withdrawal_amount || 0,
        deposit_amount: item.deposit_amount || 0,
        balance: item.balance || 0
      }));
      setTransactions(formattedData);
    } catch (error) {
      console.error('Ошибка получения транзакций:', error);
      if (error.response?.status === 401) {
        handleLogout();
      }
    }
  };

  const handleAddTransaction = async () => {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('Токен не найден');
      }
  
      const decodedToken = jwtDecode(token);
      const userId = decodedToken.id;
  
      
      if (withdrawalAmount === 0 && depositAmount === 0) {
        setSnackbar({
          open: true,
          message: 'Не указана сумма для операции (вывод или депозит)',
          severity: 'error'
        });
        return; 
      }
  
      let operationType;
      let payload = {
        category: newTransaction.category,
        description: newTransaction.description,
        withdrawal_amount: withdrawalAmount,
        deposit_amount: depositAmount,
        user: userId,
      };
  
      
      if (withdrawalAmount > 0) {
        operationType = 'withdraw';
        payload.operation_type = operationType;
      } else if (depositAmount > 0) {
        operationType = 'deposit';
        payload.operation_type = operationType;
      }
  
      
      const response = await axios.post('http://localhost:8000/transactions/', payload, {
        headers: { Authorization: `Bearer ${token}` },
      });
  
      console.log('Транзакция успешно добавлена:', response.data);
  
    } catch (error) {
      console.error('Ошибка при добавлении транзакции:', error);
      console.log('Ответ сервера:', error.response?.data);
    }
  };
  


  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await api.post('/token/', loginData);
      localStorage.setItem('access_token', response.data.access);
      localStorage.setItem('refresh_token', response.data.refresh);
      const decoded = jwtDecode(response.data.access);
      setUser(decoded);
      await fetchTransactions();
      setSnackbar({
        open: true,
        message: 'Вход выполнен успешно!',
        severity: 'success'
      });
    } catch (error) {
      setSnackbar({
        open: true,
        message: error.response?.data?.detail || 'Ошибка входа',
        severity: 'error'
      });
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setUser(null);
    setTransactions([]);
    setSnackbar({
      open: true,
      message: 'Вы вышли из системы',
      severity: 'info'
    });
  };

  if (!user) {
    return (
      <Container maxWidth="sm" sx={{ mt: 8 }}>
        <Card>
          <CardContent>
            <AppBar position="static">
              <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)} variant="fullWidth">
                <Tab label="Вход" />
                <Tab label="Регистрация" />
              </Tabs>
            </AppBar>

            {tabValue === 0 ? (
              <Box component="form" onSubmit={handleLogin} sx={{ mt: 3 }}>
                <TextField
                  label="Email"
                  type="email"
                  fullWidth
                  margin="normal"
                  value={loginData.email}
                  onChange={(e) => setLoginData({ ...loginData, email: e.target.value })}
                  required
                />
                <TextField
                  label="Пароль"
                  type="password"
                  fullWidth
                  margin="normal"
                  value={loginData.password}
                  onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
                  required
                />
                <Button type="submit" variant="contained" fullWidth sx={{ mt: 2 }}>Войти</Button>
              </Box>
            ) : (
              <Box component="form" onSubmit={handleRegister} sx={{ mt: 3 }}>
                <TextField
                  label="Имя пользователя"
                  fullWidth
                  margin="normal"
                  value={authData.username}
                  onChange={(e) => setAuthData({ ...authData, username: e.target.value })}
                  required
                />
                <TextField
                  label="Email"
                  type="email"
                  fullWidth
                  margin="normal"
                  value={authData.email}
                  onChange={(e) => setAuthData({ ...authData, email: e.target.value })}
                  required
                />
                <TextField
                  label="Пароль"
                  type="password"
                  fullWidth
                  margin="normal"
                  value={authData.password}
                  onChange={(e) => setAuthData({ ...authData, password: e.target.value })}
                  required
                />
                <TextField
                  label="Подтвердите пароль"
                  type="password"
                  fullWidth
                  margin="normal"
                  value={authData.password2}
                  onChange={(e) => setAuthData({ ...authData, password2: e.target.value })}
                  required
                />
                <Button type="submit" variant="contained" fullWidth sx={{ mt: 2 }}>Зарегистрироваться</Button>
              </Box>
            )}
          </CardContent>
        </Card>
      </Container>
    );
  }

  const categories = ['Все', ...new Set(transactions.map(t => t.category))];

  return (
    <div className={styles.container}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">Финансовый помощник</Typography>
        <Button variant="outlined" color="error" onClick={handleLogout}>Выйти ({user.username})</Button>
      </Box>

      <Box sx={{ mb: 3, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
        {categories.map(category => (
          <Button
            key={category}
            variant={selectedCategory === category ? 'contained' : 'outlined'}
            onClick={() => setSelectedCategory(category)}
          >
            {category}
          </Button>
        ))}
      </Box>

      <Box sx={{ height: 500, width: '100%', mb: 3 }}>
        <DataGrid
          rows={filteredTransactions}
          columns={columns}
          getRowId={(row) => row.id}
          initialState={{ 
            pagination: { paginationModel: { pageSize: 10 } },
            sorting: { sortModel: [{ field: 'date', sort: 'desc' }] }
          }}
          pageSizeOptions={[10, 25, 50]}
          disableRowSelectionOnClick
          sx={{
            [`& .${gridClasses.row}`]: {
              bgcolor: (theme) => theme.palette.mode === 'light' ? grey[50] : grey[900],
            },
          }}
        />
      </Box>

      <div className={styles.debug}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Typography variant="h6">Debug</Typography>
          <Switch checked={debugMode} onChange={(e) => setDebugMode(e.target.checked)} color="success" sx={{ ml: 1 }} />
        </Box>

        {debugMode && (
          <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>Добавить транзакцию</Typography>
            <TextField
              label="Категория"
              fullWidth
              margin="normal"
              value={newTransaction.category}
              onChange={(e) => setNewTransaction({ ...newTransaction, category: e.target.value })}
              required
            />

            <TextField
              label="Описание"
              fullWidth
              margin="normal"
              value={newTransaction.description}
              onChange={(e) => setNewTransaction({ ...newTransaction, description: e.target.value })}
              required
            />

            <TextField
              label="Сумма снятия"
              type="number"
              fullWidth
              margin="normal"
              value={newTransaction.withdrawal_amount}
              onChange={(e) => { 
                setNewTransaction({
                ...newTransaction, 
                withdrawal_amount: e.target.value ? Number(e.target.value) : 0,  
                deposit_amount: 0,  
                operation_type: e.target.value > 0 ? 'withdrawal' : newTransaction.operation_type,  
              });
              setWithdrawalAmount(e.target.value ? (e.target.valueAsNumber) : 0);
            }}
              disabled={newTransaction.operation_type === 'deposit'}  
            />

            <TextField
              label="Сумма пополнения"
              type="number"
              fullWidth
              margin="normal"
              value={newTransaction.deposit_amount}
              onChange={(e) => { 
                setNewTransaction({
                  ...newTransaction,
                  deposit_amount: e.target.value ? (e.target.valueAsNumber) : 0,  
                  withdrawal_amount: 0,  
                  operation_type: e.target.value > 0 ? 'deposit' : newTransaction.operation_type,  
                });
                setDepositAmount(e.target.value ? (e.target.valueAsNumber) : 0);
              }}
            />

<Button variant="contained" onClick={handleAddTransaction} fullWidth sx={{ mt: 2 }}>Добавить транзакцию</Button>
          </Paper>
        )}
      </div>

      <Snackbar open={snackbar.open} autoHideDuration={6000} onClose={() => setSnackbar({ ...snackbar, open: false })}>
        <Alert onClose={() => setSnackbar({ ...snackbar, open: false })} severity={snackbar.severity}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </div>
  );
}

export default App;
