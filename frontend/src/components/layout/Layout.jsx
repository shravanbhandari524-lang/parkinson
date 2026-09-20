import { Outlet } from 'react-router-dom';
import Navbar from './Navbar.jsx';
import Footer from './Footer.jsx';

export default function Layout() {
  return (
    <div className="flex min-h-screen flex-col bg-surface">
      <Navbar />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 pb-16 pt-8 sm:px-6">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}
